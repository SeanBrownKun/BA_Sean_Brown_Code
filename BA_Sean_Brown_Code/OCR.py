from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from re import sub

import pytesseract
from pytesseract import Output

import easyocr

import keras_ocr

import matplotlib.pyplot as plt

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from time import perf_counter


class OCR:
    """This class stores all OCR systems and is responsible for running
    their predictions. """

    def __init__(self):
        a = perf_counter()
        self.pytesseract = pytesseract.image_to_string
        b = perf_counter()
        self.easyocr = easyocr.Reader(["en"])
        c = perf_counter()
        self.keras_pipeline = keras_ocr.pipeline.Pipeline()
        d = perf_counter()
        # self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        # self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten", max_length=50)
        e = perf_counter()
        print(f"load pytesseract: {b - a}")
        print(f"load easyocr:     {c - b}")
        print(f"load keras:       {d - c}")
        print(f"load troc:        {e - d}")
        print("OCR systems loaded!")
        # languages for pytesseract: deu, fra, eng
        # languages for EasyOCR: de, fr, en

    def pytesseract_i2t(self, img, crop_tuple=None):
        """With the image file and paragraph coordinates (a tuple),
        it will run pytesseract OCR over the paragraph."""
        before = perf_counter()
        if type(img) == str:
            img = Image.open(img)
        if crop_tuple:
            img = img.crop(crop_tuple)
        img = np.array(img)

        text = self.pytesseract(img)
        text = sub("\n", " ", text)
        after = perf_counter()
        # print(f"pytesseract predict: {after - before}")
        return text

    def easyocr_i2t(self, img, crop_tuple=None):
        """With the image file and paragraph coordinates (a tuple),
        it will run EasyOCR over the paragraph."""
        before = perf_counter()
        if type(img) == str:
            img = Image.open(img)
        if crop_tuple:
            img = img.crop(crop_tuple)
        img = np.array(img)

        result = self.easyocr.readtext(img)
        text = " ".join([detection[1] for detection in result])
        after = perf_counter()
        # print(f"easyocr predict: {after - before}")
        return text

    def kerasocr_i2t(self, img, crop_tuple=None):
        """Another OCR system that was not used in the experiment."""
        before = perf_counter()
        if type(img) == str:
            img = Image.open(img)
        if crop_tuple:
            img = img.crop(crop_tuple)
        img = np.array(img)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        print(img)
        print(img.shape)
        imgs = [keras_ocr.tools.read(img)]
        text = self.keras_pipeline.recognize(imgs)[0]

        text_str = " ".join(word[0] for word in text)
        after = perf_counter()
        print(f"kerasocr predict: {after - before}")
        return text_str

    # Another OCR system, that was not used in the experiment.
    # It is commented out because it takes a long time to load
    # every time the OCR class is instantiated.
    """def trocr_i2t(self, img, crop_tuple=None):
        before = perf_counter()
        if type(img) == str:
            img = Image.open(img)
        if crop_tuple:
            img = img.crop(crop_tuple)
            img.show()
        img = np.array(img)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        pixel_values = self.trocr_processor(img, return_tensors="pt").pixel_values
        generated_ids = self.trocr_model.generate(pixel_values)
        generated_text = self.trocr_processor.batch_decode(generated_ids)[0]
        after = perf_counter()
        print(f"trocr predict: {after - before}")
        return generated_text"""

    def prediction_from_all_ocr_models(self, img, crop_tuple=None, skip_keras=False):
        """This method runs predictions over all OCR systems and returns
        the results in a list."""
        if skip_keras:
            return [self.pytesseract_i2t(img, crop_tuple),
                    self.easyocr_i2t(img, crop_tuple)]
        else:
            return [self.pytesseract_i2t(img, crop_tuple),
                    self.easyocr_i2t(img, crop_tuple),
                    self.kerasocr_i2t(img, crop_tuple)]

    @staticmethod
    def show(img, crop_tuple=None):
        """Static method that shows an image or paragraph."""
        img = Image.open(img)
        if crop_tuple:
            img = img.crop(crop_tuple)
        img = np.array(img)
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    print(pytesseract.__version__)
    print(easyocr.__version__)

