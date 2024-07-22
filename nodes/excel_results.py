from io import BytesIO
from pathlib import Path
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Border, Side, Font
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.cell.cell import Cell
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.utils import quote_sheetname, absolute_coordinate, get_column_letter
import polars as pl

from nodes.actions.action import ActionNode
from nodes.constants import FORECAST_COLUMN, IMPACT_COLUMN, IMPACT_GROUP, VALUE_COLUMN, YEAR_COLUMN
from nodes.context import Context
from nodes.dimensions import Dimension
from nodes.node import Node


DATA_SHEET_NAME = 'Data'
PARAM_SHEET_NAME = 'Parameters'


NODES = '''
net_emissions
transport_emissions
freight_transport_emissions
waste_emissions
forestation_emissions
electricity_production_emissions
building_emissions
vehicle_kilometres
freight_transport_vehicle_kilometres
building_heat_energy_use
consumer_electricity_use
collected_waste
'''.strip().splitlines()

def _convert_to_camel_case(input_string: str):
    # Split the input string into words
    words = input_string.split()

    # Capitalize the first letter of each word
    capitalized_words = [word.capitalize() for word in words]

    # Join the words without spaces
    camel_case_string = ''.join(capitalized_words)

    return camel_case_string


def _dim_to_col(dim: Dimension):
    return _convert_to_camel_case(str(dim.label))

FIXED_COLUMNS = [
    'Node',
    'Year',
    'Unit',
    'Value',
    'BaselineValue',
    'Forecast',
]


def _output_node(wb: Workbook, sheet: Worksheet, node: Node, dim_ids: list[str], actions: list[ActionNode]):
    df = node.get_output_pl()
    if df.dim_ids:
        df = df.with_columns([pl.col(dim_id).cast(pl.String) for dim_id in df.dim_ids])
    df = df.sort(by=[YEAR_COLUMN, *df.dim_ids])
    bdf = node.baseline_values
    assert bdf is not None
    bdf = bdf.with_columns(pl.col(VALUE_COLUMN).alias('BaselineValue')).drop(VALUE_COLUMN)
    df = df.paths.join_over_index(bdf, how='left', index_from='left')
    col_map = {
        'Node': pl.lit(node.id),
        'Year': pl.col(YEAR_COLUMN),
        'Unit': pl.lit(str(node.unit)),
        'Value': pl.col(VALUE_COLUMN),
        'BaselineValue': pl.col('BaselineValue'),
        'Forecast': pl.col(FORECAST_COLUMN)
    }
    cols = [col_map[col_id].alias(col_id) for col_id in FIXED_COLUMNS]
    for dim_id in dim_ids:
        if dim_id in df.dim_ids:
            cols.append(pl.col(dim_id))
        else:
            cols.append(pl.lit(None).alias(dim_id))

    for act in actions:
        adf = act.compute_impact(node)
        act_col = 'Impact_%s' % act.id
        adf = adf.filter(pl.col(IMPACT_COLUMN) == IMPACT_GROUP).drop(IMPACT_COLUMN).rename({VALUE_COLUMN: act_col})
        df = df.paths.join_over_index(adf, how='left', index_from='left')
        cols.append(pl.col(act_col))

    start_row = sheet.max_row + 1

    df = df.select(cols)
    for row in df.iter_rows():
        sheet.append(row)
    end_row = sheet.max_row

    start_col = get_column_letter(1)
    end_col = get_column_letter(len(row))
    range = absolute_coordinate('%s%s:%s%s' % (start_col, start_row, end_col, end_row))
    ref = '%s!%s' % (quote_sheetname(sheet.title), range)
    defn = DefinedName(node.id, attr_text=ref)
    wb.defined_names[node.id] = defn


def _add_param_sheet(context: Context, wb: Workbook):
    ps: Worksheet = wb.create_sheet(PARAM_SHEET_NAME)
    ps.append(['Parameter', 'Value'])

    max_name_length: int = 20

    def add_param_value(name: str, value: float | int | str | None, named_range_name: str | None = None):
        nonlocal max_name_length

        if value is None:
            return
        range_name = named_range_name or _convert_to_camel_case(name)
        ps.append([name, value])
        def_range = '$B$%d' % ps.max_row
        defn = DefinedName(range_name, attr_text='%s!%s' % (quote_sheetname(ps.title), def_range))
        wb.defined_names[range_name] = defn
        if len(name) > max_name_length:
            max_name_length = len(name)

    add_param_value('Baseline year', context.instance.maximum_historical_year, 'BaselineYear')
    add_param_value('Target year', context.model_end_year)
    ps.column_dimensions['A'].width = max_name_length


def create_result_excel(context: Context, existing_wb: Path | None = None):
    node_ids = NODES
    context.generate_baseline_values()
    all_dims = set()

    for node_id in node_ids:
        node = context.nodes[node_id]
        for dim in node.output_dimensions.values():
            if dim.id not in all_dims:
                all_dims.add(dim.id)

    created_sheet_names = (DATA_SHEET_NAME, PARAM_SHEET_NAME)
    if existing_wb:
        wb = load_workbook(str(existing_wb), rich_text=True)
        to_remove: set[str] = set()
        for name, defn in wb.defined_names.items():
            for sheet_name, _ in defn.destinations:
                if sheet_name in created_sheet_names:
                    to_remove.add(name)
                    break
        for def_name in to_remove:
            del wb.defined_names[def_name]
    else:
        # Create a new workbook
        wb = Workbook()
        del wb[wb.sheetnames[0]]

    for sn in created_sheet_names:
        if sn in wb:
            del wb[sn]
        pass

    ds: Worksheet = wb.create_sheet(DATA_SHEET_NAME)

    dims = [context.dimensions[dim_id] for dim_id in all_dims]
    cols = FIXED_COLUMNS + [_dim_to_col(dim) for dim in dims]
    actions = context.get_actions()[1:]
    for act in actions:
        cols.append('Impact_%s' % act.id)
    ds.append(cols)
    for idx, col in enumerate(cols):
        letter = get_column_letter(idx + 1)
        ref = '%s!%s' % (quote_sheetname(ds.title), '$%s:$%s' % (letter, letter))
        defn = DefinedName(col, attr_text=ref)
        wb.defined_names[col] = defn

        cell: Cell = ds['%s1' % letter]
        cell.border = Border(bottom=Side(style='thin', color='222222'))
        cell.font = Font(bold=True)

    dim_ids = [dim.id for dim in dims]
    for node_id in node_ids:
        node = context.nodes[node_id]
        _output_node(wb, ds, node, dim_ids=dim_ids, actions=actions)
    ds.freeze_panes = ds['B2']

    ds.column_dimensions['A'].width = 20
    #for col in range(ds.min_column, ds.max_column + 1):
    #    ds.column_dimensions[get_column_letter(col)].bestFit = True
    #    ds.column_dimensions[get_column_letter(col)].auto_size = True

    _add_param_sheet(context, wb)

    buffer = BytesIO()
    wb.save(buffer)
    return buffer
