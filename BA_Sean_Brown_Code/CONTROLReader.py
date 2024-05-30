from PIL import Image, ImageDraw, ImageFont
from math import floor
from pipeline.OCR import OCR


class CONTROLReader:
    """This class generated:
        - the easy corpus
        - the pytesseract OCR and EasyOCR output
        - two prompt files"""

    def __init__(self):
        self.sentences = []

    def load_sentences(self):
        """This method loads the ground truth sentences into the self.sentences list"""
        with open("CLEAR_CORPUS/sentences", "r", encoding='utf-8') as file:
            for line in file:
                self.sentences.append(line.strip())

    def format_text(self, text):
        """This method makes sure that an input doesn't go beyond the
        image bound when it is too long, but inserts a newline. See the
        example image in my paper."""
        num_lines = floor(len(text) / 30)
        lines = []
        for i in range(num_lines + 1):
            lines.append(text[:30])
            if len(text) > 30:
                text = text[30:]
            else:
                break
        return "\n".join(lines)

    def return_image_and_text(self, sent_idx):
        """This method creates a black-on-white image with the sentence as input
        and both an image and the formatted text as output."""
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        text = self.sentences[sent_idx]
        text = self.format_text(text)
        text_width, text_height = draw.textsize(text, font=font)
        position = ((img.width - text_width) // 2, (img.height - text_height) // 2)
        draw.text(position, text, font=font, fill='black')
        return img, text

    def load_predictions(self, ocr, from_file=True):
        """This class returns the previously generated OCR outputs
        or makes new outputs if they haven't been generated yet."""
        self.load_sentences()
        predictions = []
        if from_file:
            pytesseract = open("CLEAR_CORPUS/pytesseract_output", "r", encoding='utf-8')
            easyocr = open("CLEAR_CORPUS/easyocr_output", "r", encoding='utf-8')
            for pyt, eas in zip(pytesseract, easyocr):
                predictions.append([pyt, eas])
        else:
            for sent_idx in range(len(self.sentences)):
                print(f"sentences ocr'd: {sent_idx} / {len(self.sentences)}")
                image, text = self.return_image_and_text(sent_idx)
                preds = ocr.prediction_from_all_ocr_models(image, skip_keras=True)
                predictions.append(preds)
                print(preds)
                image.show()
                input()

            self.make_prompts_file(predictions)
        return self.sentences.copy(), predictions

    def make_prompts_file(self, predictions):
        """This method makes the prompt file, by inserting every OCR prediction
        into the prompt template."""
        best_file = open("CLEAR_CORPUS/prompts_best2", "w", encoding='utf-8')
        all_file = open("CLEAR_CORPUS/prompts_all2", "w", encoding='utf-8')
        for preds in predictions:
            best = preds[0]
            best_prompt = ("My OCR system gave me this output:", "Please correct it. It's very important that you only correct the output. Do not make further comments!")
            all_prompt = ("My two OCR systems gave me these outputs:", "Please guess the original input. Only give me the one phrase. Do not make further comments!")
            p_best = f"'{best_prompt[0]} '{best}' {best_prompt[1]}'"
            interjection = "' and '"
            p_all = f"'{all_prompt[0]} '{interjection.join(preds)}' {all_prompt[1]}'"
            best_file.write(f"{p_best}\n")
            all_file.write(f"{p_all}\n")
        best_file.close()
        all_file.close()

        """It also writes the OCR output files from the previously generated OCR predictions."""
        # also make output files
        pytesseract = open("CLEAR_CORPUS/pytesseract_output", "w", encoding='utf-8')
        easyocr = open("CLEAR_CORPUS/easyocr_output", "w", encoding='utf-8')
        for preds in predictions:
            pytesseract.write(f"{preds[0]}\n")
            easyocr.write(f"{preds[1]}\n")
        pytesseract.close()
        easyocr.close()


if __name__ == '__main__':
    import PIL
    print(PIL.__version__)
    input()
    ocr = OCR()
    cr = CONTROLReader()
    o, p = cr.load_predictions(ocr, from_file=True)
    print(o)
    print(p)


