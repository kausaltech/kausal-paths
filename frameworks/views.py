from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden, HttpResponseServerError
from django.shortcuts import get_object_or_404

import sentry_sdk
from loguru import logger as base_logger

from frameworks.models import FrameworkConfig
from nodes.excel_results import ExportNotSupportedError, InstanceResultExcel

if TYPE_CHECKING:
    from paths.types import PathsRequest


def create_result_file(_request: PathsRequest, fwc_id: int, token: str) -> HttpResponse:
    logger = base_logger.bind(fwc_id=fwc_id)
    logger.info('Request to create results excel for framework config')

    fwc = get_object_or_404(FrameworkConfig, id=fwc_id)
    if token != fwc.token:
        return HttpResponseForbidden('Invalid token')

    ic = fwc.instance_config
    instance = ic.get_instance()
    logger = instance.log

    logger.info('Creating result excel for framework config: %s' % str(fwc))
    try:
        context = instance.context
        with instance.context.run():
            buffer = InstanceResultExcel.create_for_instance(ic, existing_wb=None, context=context)
    except ExportNotSupportedError as err:
        sentry_sdk.capture_exception(err)
        logger.exception('Export not supported for instance')
        return HttpResponseBadRequest('Export not supported')
    except Exception as err:
        sentry_sdk.capture_exception(err)
        logger.exception('Error creating result .xlsx for instance')
        return HttpResponseServerError('Unable to create export')
    data = buffer.getvalue()
    logger.info('Excel generated, size %d bytes' % len(data))

    # Include the current date and time in the filename in YYYYmmdd-HHMM format
    current_time = datetime.now(tz=UTC).strftime('%Y%m%d-%H%M')
    ic = fwc.instance_config
    fn = f'{ic.identifier}_results_{current_time}.xlsx'
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=%s' % fn

    # Write the workbook data to the response
    response.write(data)

    return response
