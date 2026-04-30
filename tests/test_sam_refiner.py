import unittest

import numpy as np

import tools.review.labeller.sam_refiner as sam_refiner
from tools.review.labeller.sam_refiner import Sam3RoiRefiner


class _FakeTensor:
    def __init__(self, value):
        self.value = np.array(value)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, idx):
        item = self.value[idx]
        if np.isscalar(item):
            return _FakeScalar(item)
        return _FakeTensor(item)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.value)

    def argmax(self):
        return _FakeScalar(int(np.argmax(self.value)))


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.value)


class _FakeProcessor:
    def set_image(self, _image):
        return object()

    def set_text_prompt(self, *, state, prompt):
        del state, prompt
        return {
            "masks": _FakeTensor([[[1, 1], [1, 1]]]),
            "boxes": _FakeTensor([[0, 0, 1, 1]]),
            "scores": _FakeTensor([0.9]),
        }


class SamRefinerLoadCachingTest(unittest.TestCase):
    def test_refine_loads_only_once_after_model_init(self):
        original_is_sam3_available = sam_refiner._is_sam3_available
        original_is_hf_available = sam_refiner._is_hf_available
        original_min_mask_area_px = sam_refiner.SAM_MIN_MASK_AREA_PX
        try:
            sam_refiner._is_sam3_available = lambda: True
            sam_refiner._is_hf_available = lambda: True
            sam_refiner.SAM_MIN_MASK_AREA_PX = 1

            refiner = Sam3RoiRefiner(device="cuda:0")
            refiner._gpu_ready = lambda: True
            load_calls = {"count": 0}

            def fake_load():
                load_calls["count"] += 1
                refiner.model = object()
                refiner.processor = _FakeProcessor()
                return refiner

            refiner.load = fake_load

            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            rois = [{"kind": "grounding_anchor_region", "bbox_xyxy": [1, 1, 6, 6]}]

            first = refiner.refine(frame, rois)
            second = refiner.refine(frame, rois)

            self.assertEqual(first.status, "ready")
            self.assertEqual(second.status, "ready")
            self.assertEqual(load_calls["count"], 1)
            self.assertEqual(len(first.proposals), 1)
            self.assertEqual(len(second.proposals), 1)
        finally:
            sam_refiner._is_sam3_available = original_is_sam3_available
            sam_refiner._is_hf_available = original_is_hf_available
            sam_refiner.SAM_MIN_MASK_AREA_PX = original_min_mask_area_px


if __name__ == "__main__":
    unittest.main()
