import base64
import io
import logging
from typing import Any, Iterable

from docling_core.types.doc import BoundingBox, CoordOrigin
from PIL.Image import Image
from pyrate_limiter import Duration, Limiter, RequestRate
from requests_ratelimiter import LimiterSession

from docling.datamodel.base_models import OcrCell, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import VisionOcrOptions
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class VisionOcrModel(BaseOcrModel):
    def __init__(
        self,
        enabled: bool,
        options: VisionOcrOptions,
    ):
        super().__init__(enabled=enabled, options=options)
        self.options: VisionOcrOptions
        self.scale = 3  # multiplier for 72 dpi == 216 dpi.

        self.limiter_session = LimiterSession(limiter=self.options.limiter)

    def _recognise_image(self, image: Image) -> Any:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_content = base64.b64encode(buffered.getvalue()).decode("utf-8")

        url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {:s}".format(self.options.iam_token),
            "x-folder-id": self.options.folder_id,
            "x-data-logging-enabled": "true",
            "model": "page",
        }

        data = {
            "content": image_content,
            "languageCodes": self.options.lang,
            "mimeType:": "image/png",
        }

        response = self.limiter_session.post(url, headers=headers, json=data)
        return response.json()

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "ocr"):
                    ocr_rects = self.get_ocr_rects(page)

                    all_ocr_cells = []
                    for ocr_rect in ocr_rects:
                        # Skip zero area boxes
                        if ocr_rect.area() == 0:
                            continue

                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )

                        result = self._recognise_image(high_res_image)
                        if "error" in result:
                            raise RuntimeError(result["error"]["message"])

                        cells = []
                        text_annotation = result["result"]["textAnnotation"]

                        blocks = text_annotation.get("blocks", [])
                        for ix, block in enumerate(blocks):
                            text = "\n".join(line["text"] for line in block["lines"])

                            vertices = block["boundingBox"]["vertices"]
                            min_x = int(vertices[0]["x"])
                            min_y = int(vertices[0]["y"])
                            max_x = int(vertices[2]["x"])
                            max_y = int(vertices[2]["y"])
                            bbox = BoundingBox.from_tuple(
                                coord=(
                                    (min_x / self.scale) + ocr_rect.l,
                                    (min_y / self.scale) + ocr_rect.t,
                                    (max_x / self.scale) + ocr_rect.l,
                                    (max_y / self.scale) + ocr_rect.t,
                                ),
                                origin=CoordOrigin.TOPLEFT,
                            )

                            cells.append(
                                OcrCell(
                                    id=ix,
                                    text=text,
                                    confidence=1.0,
                                    bbox=bbox,
                                )
                            )

                        all_ocr_cells.extend(cells)

                    # Post-process the cells
                    page.cells = self.post_process_cells(all_ocr_cells, page.cells)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page
