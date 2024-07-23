from datetime import datetime

from django.http import HttpResponse, HttpResponseForbidden, HttpResponseNotFound
from django.shortcuts import get_object_or_404
from loguru import logger

from frameworks.models import FrameworkConfig
from nodes.excel_results import create_result_excel

from paths.types import PathsRequest


def create_result_file(request: PathsRequest, fwc_id: int, token: str):
    logger.info("Request to create results excel for framework config %d" % fwc_id)

    fwc = get_object_or_404(FrameworkConfig, id=fwc_id)
    if token != fwc.token:
        return HttpResponseForbidden('Invalid token')

    ic = fwc.instance_config
    fw = fwc.framework

    # Create the workbook using create_result_excel method
    context = ic.get_instance().context
    if not fw.result_excel_url or not fw.result_excel_node_ids:
        return HttpResponseNotFound("Framework doesn't support generating result files")

    logger.info("Creating result excel for framework config: %s" % str(fwc))
    buffer = create_result_excel(context, fw.result_excel_url, fw.result_excel_node_ids)
    data = buffer.getvalue()
    logger.info("Excel generated, size %d bytes" % len(data))


    # Include the current date and time in the filename in YYYYmmdd-HHMM format
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    ic = fwc.instance_config
    fn = f'{ic.identifier}_results_{current_time}.xlsx'
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=%s' % fn

    # Write the workbook data to the response
    response.write(data)

    return response
