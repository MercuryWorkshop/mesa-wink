# SPDX-License-Identifier: MIT
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw
from compare import compare_images, compare_directories


class ImageOracleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def images(self, a, b):
        a.save(self.root / "a.png")
        b.save(self.root / "b.png")
        return self.root / "a.png", self.root / "b.png"

    def test_single_blue_channel_corruption(self):
        a, b = self.images(Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32), (0, 0, 16)))
        self.assertFalse(compare_images(a, b)["pass"])

    def test_thin_missing_draw_not_hidden_by_erosion(self):
        a = Image.new("RGB", (100, 100))
        b = a.copy()
        ImageDraw.Draw(b).line((50, 0, 50, 99), fill="white")
        result = compare_images(*self.images(a, b))
        self.assertEqual(result["interior"], 0)
        self.assertFalse(result["pass"])

    def test_small_thin_feature_region(self):
        a = Image.new("RGB", (100, 100))
        b = a.copy()
        b.putpixel((50, 50), (255, 255, 255))
        policy = {"regions": [{"name": "text", "box": [49, 49, 52, 52]}]}
        self.assertFalse(compare_images(*self.images(a, b), policy)["pass"])

    def test_vertical_flip(self):
        a = Image.new("RGB", (32, 32))
        ImageDraw.Draw(a).rectangle((0, 0, 31, 10), fill="red")
        self.assertFalse(compare_images(*self.images(a, a.transpose(Image.Transpose.FLIP_TOP_BOTTOM)))["pass"])

    def test_match_and_tolerance(self):
        self.assertTrue(compare_images(*self.images(Image.new("RGB", (16, 16), (20, 30, 40)),
                                                    Image.new("RGB", (16, 16), (25, 30, 40))))["pass"])

    def test_missing_and_size_mismatch(self):
        self.assertFalse(compare_directories(self.root, self.root)[0]["pass"])
        self.assertFalse(compare_directories(self.root, self.root, expected=["frame.png"])[0]["pass"])
        self.assertFalse(compare_images(*self.images(Image.new("RGB", (2, 2)), Image.new("RGB", (3, 2))))["pass"])

    def test_alpha_opt_in(self):
        args = self.images(Image.new("RGBA", (16, 16), (10, 20, 30, 0)),
                           Image.new("RGBA", (16, 16), (10, 20, 30, 255)))
        self.assertTrue(compare_images(*args)["pass"])
        self.assertFalse(compare_images(*args, {"alpha": True})["pass"])


if __name__ == "__main__":
    unittest.main()
