from pipeline.OCR import OCR
from pipeline.NZZReader import NZZReader
from pipeline.CONTROLReader import CONTROLReader
from pipeline.Evaluator import Evaluator
import matplotlib.pyplot as plt
import numpy as np
import scipy
from re import sub
from random import randint


class Master:
    """This class is responsible for visualizing the results."""

    def __init__(self):
        self.ocr = OCR()
        self.eval = Evaluator()
        self.document = CONTROLReader()
        # self.nzz = NZZReader()

        self.original, self.predictions = self.document.load_predictions(self.ocr)

    def evaluate_ocr_outputs(self, pyt, easy, org, word_level=False):
        """This method evaluates the two OCR outputs from the generated files
        and visualizes them in a graph. It also prints the p-values and means
        to the console.

        It also handles generating OCR outputs and prompt files for the NZZ corpus."""

        pyt_scores, eas_scores = [], []
        pyt = [x.strip() for x in open(pyt, "r", encoding='utf-8').readlines()]
        easy = [x.strip() for x in open(easy, "r", encoding='utf-8').readlines()]
        org = [x.strip() for x in open(org, "r", encoding='utf-8').readlines()]
        for p, e, o in zip(pyt, easy, org):
            p_result = self.eval.evaluate_one_example(p, o, word_level=word_level)
            e_result = self.eval.evaluate_one_example(e, o, word_level=word_level)
            pyt_scores.append(- p_result[1] / len(o))
            eas_scores.append(- e_result[1] / len(o))
        pyt_scores = np.array(pyt_scores)
        eas_scores = np.array(eas_scores)
        x_axis = np.arange(1, len(pyt_scores)+1)

        # for an oracle model:
        # optimal = np.where(pyt_scores > eas_scores, pyt_scores, eas_scores)

        fig, axs = plt.subplots(1, 2)
        axs[1].boxplot([pyt_scores, eas_scores, 0])
        axs[1].set_xticklabels(["Pytesseract OCR", "EasyOCR", "GT"])

        axs[0].scatter(x_axis, pyt_scores, color='orange', label='Pytesseract OCR')
        axs[0].scatter(x_axis, eas_scores, color='blue', label='EasyOCR')
        # plt.scatter(x_axis, optimal, color='green', label='Ground Truth', marker="x")
        axs[0].plot(x_axis, np.full(len(x_axis), np.median(pyt_scores)), color='orange', label='Pytesseract OCR median')
        axs[0].plot(x_axis, np.full(len(x_axis), np.median(eas_scores)), color='blue', label='EasyOCR median')
        axs[0].plot(x_axis, np.full(len(x_axis), 0), color='green', label='Ground Truth')
        # plt.plot(x_axis, np.full(len(x_axis), optimal.mean()), color='green')

        if word_level:
            plt.suptitle("Pytesseract OCR vs. EasyOCR (token level)")
            axs[0].axis(ymin=-.25, ymax=0.05)
            axs[1].axis(ymin=-.25, ymax=0.05)
        else:
            plt.suptitle("Pytesseract OCR vs. EasyOCR (character level)")
            axs[0].axis(ymin=-.8, ymax=0.05)
            axs[1].axis(ymin=-.8, ymax=0.05)

        result = scipy.stats.mannwhitneyu(pyt_scores, eas_scores)
        print(f"Mannwhitney-test for the two ocr outputs:")
        print(f"pvalue {result.pvalue}")

        # optimal_result = scipy.stats.mannwhitneyu(pyt_scores, optimal)
        # print(f"Mannwhitney-test for optimal:")
        # print(f"pvalue {optimal_result.pvalue}")

        axs[0].set_xlabel("Examples")
        axs[0].set_ylabel("Correctness")
        axs[1].set_xlabel("Examples")
        axs[1].set_ylabel("Correctness")
        axs[0].legend()
        axs[1].legend()
        print(f"pytesseract average score: {pyt_scores.mean()}")
        print(f"easyocr average score:     {eas_scores.mean()}")
        plt.show()

    def evaluate_llm_outputs(self, best, all, original, ocr_comparison, word_level=False):
        """This method evaluates the two versions of an LLM - one with only pytesseract OCR
        as input and the other with both OCRs as input - from the generated files
        and visualizes them in a graph. It also prints the p-values and means
        to the console."""
        best = [x.strip() for x in open(best, "r", encoding='utf-8').readlines()]
        all = [x.strip() for x in open(all, "r", encoding='utf-8').readlines()]
        original = [x.strip() for x in open(original, "r", encoding='utf-8').readlines()]
        ocr_comparison = [x.strip() for x in open(ocr_comparison, "r", encoding='utf-8').readlines()]
        best_scores = []
        all_scores = []
        pyt_ocr_scores = []
        for b, a, ocr, org in zip(best, all, ocr_comparison, original):
            best_result = self.eval.evaluate_one_example(b, org, word_level=word_level)
            all_result = self.eval.evaluate_one_example(a, org, word_level=word_level)
            pyt_ocr_result = self.eval.evaluate_one_example(ocr, org, word_level=word_level)
            best_scores.append(- best_result[1] / len(org))
            all_scores.append(- all_result[1] / len(org))
            pyt_ocr_scores.append(- pyt_ocr_result[1] / len(org))
        best_scores = np.array(best_scores)
        all_scores = np.array(all_scores)
        pyt_ocr_scores = np.array(pyt_ocr_scores)
        x_axis = np.arange(1, len(best_scores) + 1)

        fig, axs = plt.subplots(1, 2)
        axs[1].boxplot([best_scores, all_scores, pyt_ocr_scores])
        axs[1].set_xticklabels(["Only one input", "Both inputs", "OCR output"])

        axs[0].scatter(x_axis, best_scores, color='green', label='Using only pytesseract output')
        axs[0].scatter(x_axis, all_scores, color='orange', label="Using both OCR systems' outputs", marker="o")
        axs[0].scatter(x_axis, pyt_ocr_scores, color=(0, 1, 0.8, 0.5), label="Original OCR output", marker="x")
        axs[0].plot(x_axis, np.full(len(x_axis), np.median(best_scores)), color='green', label='Median only one input')
        axs[0].plot(x_axis, np.full(len(x_axis), np.median(all_scores)), color='orange', label='Median both inputs')
        axs[0].plot(x_axis, np.full(len(x_axis), np.median(pyt_ocr_scores)), color=(0, .5, 1, 1), label='Median ocr output')

        if word_level:
            plt.suptitle("Only one input vs. both inputs vs. OCR output (token-level)")
            axs[0].axis(ymin=-.2, ymax=0.05)
            axs[1].axis(ymin=-.2, ymax=0.05)
        else:
            plt.suptitle("Only one input vs. both inputs vs. OCR output (character-level)")
            axs[0].axis(ymin=-1, ymax=0.03)
            axs[1].axis(ymin=-1, ymax=0.03)
        axs[0].set_xlabel("Examples")
        axs[0].set_ylabel("Correctness")
        axs[1].set_xlabel("Examples")
        axs[1].set_ylabel("Correctness")
        axs[0].legend()
        axs[1].legend()
        print(len(best_scores))
        print(len(all_scores))
        print("--- averages")
        print(f"only pytesseract average score: {best_scores.mean()}")
        print(f"both ocr systems average score: {all_scores.mean()}")
        print(f"ocr output average score:       {pyt_ocr_scores.mean()}")
        print("--- medians")
        print(f"only pytesseract median score: {np.median(best_scores)}")
        print(f"both ocr systems median score: {np.median(all_scores)}")
        print(f"ocr output average score:      {np.median(pyt_ocr_scores)}")

        result = scipy.stats.mannwhitneyu(best_scores, all_scores)
        result2 = scipy.stats.mannwhitneyu(all_scores, pyt_ocr_scores)
        result3 = scipy.stats.mannwhitneyu(best_scores, pyt_ocr_scores)
        print(f"Mannwhitney-test for the two chatgpt3 outputs:")
        print(f"pvalue {result.pvalue}")
        print(f"Mannwhitney-test comparing using both ocr outputs to ocr output:")
        print(f"pvalue {result2.pvalue}")
        print(f"Mannwhitney-test comparing using only one ocr output to ocr output:")
        print(f"pvalue {result3.pvalue}")
        plt.show()

    def compare_two_llm_bests(self, llm, chatgpt, original, ocr_comparison, word_level=False):
        """This method compares the Fastchat-t5 and ChatGPT3.5 models that were given only
        the "best" OCR prediction (pytesseract OCR) - hence "bests" - from the generated files
        and visualizes them in a graph. It also prints the p-values and means
        to the console."""
        best = [x.strip() for x in open(llm, "r", encoding='utf-8').readlines()]
        all = [x.strip() for x in open(chatgpt, "r", encoding='utf-8').readlines()]
        original = [x.strip() for x in open(original, "r", encoding='utf-8').readlines()]
        ocr_comparison = [x.strip() for x in open(ocr_comparison, "r", encoding='utf-8').readlines()]
        best_scores = []
        all_scores = []
        pyt_ocr_scores = []
        for b, a, ocr, org in zip(best, all, ocr_comparison, original):
            best_result = self.eval.evaluate_one_example(b, org, word_level=word_level)
            all_result = self.eval.evaluate_one_example(a, org, word_level=word_level)
            pyt_ocr_result = self.eval.evaluate_one_example(ocr, org, word_level=word_level)
            best_scores.append(- best_result[1] / len(org))
            all_scores.append(- all_result[1] / len(org))
            pyt_ocr_scores.append(- pyt_ocr_result[1] / len(org))
        best_scores = np.array(best_scores)
        all_scores = np.array(all_scores)
        pyt_ocr_scores = np.array(pyt_ocr_scores)
        x_axis = np.arange(1, len(best_scores) + 1)

        fig, axs = plt.subplots(1, 2)
        axs[1].boxplot([best_scores, all_scores, pyt_ocr_scores])
        axs[1].set_xticklabels(["fastchat-t5", "ChatGPT3.5", "OCR output"])

        axs[0].scatter(x_axis, best_scores, color='green', label='Using fastchat-t5')
        axs[0].scatter(x_axis, all_scores, color='orange', label="Using ChatGPT3.5", marker="o")
        axs[0].scatter(x_axis, pyt_ocr_scores, color=(0, 1, 0.8, 0.5), label="Original OCR output", marker="x")
        axs[0].plot(x_axis, np.full(len(x_axis), np.median(best_scores)), color='green', label='Median fastchat-t5')
        axs[0].plot(x_axis, np.full(len(x_axis), np.median(all_scores)), color='orange', label='Median ChatGPT3.5')
        axs[0].plot(x_axis, np.full(len(x_axis), np.median(pyt_ocr_scores)), color=(0, .5, 1, 1), label='Median ocr output')

        if word_level:
            plt.suptitle("fastchat-t5 vs. ChatGPT3.5 vs. OCR output (token-level)")
            axs[0].axis(ymin=-.2, ymax=0.05)
            axs[1].axis(ymin=-.2, ymax=0.05)
        else:
            plt.suptitle("fastchat-t5 vs. ChatGPT3.5 vs. OCR output (character-level)")
            axs[0].axis(ymin=-0.5, ymax=0.03)
            axs[1].axis(ymin=-0.5, ymax=0.03)
        axs[0].set_xlabel("Examples")
        axs[0].set_ylabel("Correctness")
        axs[1].set_xlabel("Examples")
        axs[1].set_ylabel("Correctness")
        axs[0].legend()
        axs[1].legend()
        print(len(best_scores))
        print(len(all_scores))
        print("--- averages")
        print(f"only pytesseract average score: {best_scores.mean()}")
        print(f"both ocr systems average score: {all_scores.mean()}")
        print(f"ocr output average score:       {pyt_ocr_scores.mean()}")
        print("--- medians")
        print(f"only pytesseract median score: {np.median(best_scores)}")
        print(f"both ocr systems median score: {np.median(all_scores)}")
        print(f"ocr output average score:      {np.median(pyt_ocr_scores)}")

        llm = scipy.stats.mannwhitneyu(best_scores, pyt_ocr_scores)
        chatgpt = scipy.stats.mannwhitneyu(all_scores, pyt_ocr_scores)
        print(f"Mannwhitney-test comparing fastchat-t5 to pyt ocr:")
        print(f"pvalue {llm.pvalue}")
        print(f"Mannwhitney-test comparing chatgpt to pyt ocr:")
        print(f"pvalue {chatgpt.pvalue}")
        plt.show()

    def get_ocr_outputs_for_NZZ_corpus(self, range_tuple=(1780, 1946), sample_size=200):
        """This class is used to generate the OCR outputs for the NZZ corpus."""
        entries = NZZReader.readall(range_tuple)
        pyt_collection = []
        easy_collection = []
        org = []

        idx = 0
        for i in range(sample_size):
            entry_index = randint(0, len(entries)-1)
            entry = entries.pop(entry_index)
            idx += 1
            if idx % 20 == 0:
                print(f"doing ocr for entry {idx} / {sample_size} (could be: {len(entries)})")
            img_file = f"../NZZ-black-letter-ground-truth-master/img/{entry[0]}"
            coords = entry[1]
            ground_truth = entry[3].strip()
            pytesseract_prediction = self.ocr.pytesseract_i2t(img_file, crop_tuple=coords)
            easyocr_prediction = self.ocr.easyocr_i2t(img_file, crop_tuple=coords)
            pyt_collection.append(pytesseract_prediction)
            easy_collection.append(easyocr_prediction)
            new_ground_truth = sub("\n", " ", ground_truth)
            # print(f"ground truth: {new_ground_truth}")
            # print(f"pytesseract:  {pytesseract_prediction}")
            # print(f"easyocr:      {easyocr_prediction}")
            org.append(new_ground_truth)

        with open(f"NZZ_CORPUS/sentences_{range_tuple[0]}_{range_tuple[1]}", "w", encoding='utf-8') as file:
            for sentence in org:
                file.write(f"{sentence.strip()}\n")
        with open(f"NZZ_CORPUS/pytesseract_output_{range_tuple[0]}_{range_tuple[1]}", "w", encoding='utf-8') as file:
            for pyt_out in pyt_collection:
                file.write(f"{pyt_out.strip()}\n")
        with open(f"NZZ_CORPUS/easyocr_output_{range_tuple[0]}_{range_tuple[1]}", "w", encoding='utf-8') as file:
            for easy_out in easy_collection:
                file.write(f"{easy_out.strip()}\n")


    def make_prompts_file(self):
        """This class makes the prompts for the NZZ corpus."""
        prompt_best = ("My OCR system gave me this output:", "Please correct it. It's from a German newspaper in Switzerland. It's very important that you only correct the output. Do not make further comments!")
        prompt_all = ("My two OCR systems gave me these outputs:", "Please guess the original input. It's from a German newspaper in Switzerland. Only give me the prediction. Do not make further comments!")

        pyt_file = open("NZZ_CORPUS/pytesseract_output_1780_1946", "r", encoding='utf-8')
        easy_file = open("NZZ_CORPUS/easyocr_output_1780_1946", "r", encoding='utf-8')

        best_file = open("NZZ_CORPUS/best_prompts", "w", encoding='utf-8')
        all_file = open("NZZ_CORPUS/all_prompts", "w", encoding='utf-8')

        pyt, easy = pyt_file.readlines(), easy_file.readlines()

        for p, e in zip(pyt, easy):
            best_prompt = f"{prompt_best[0]} '{p.strip()}' {prompt_best[1]}\n"
            all_prompt = f"{prompt_all[0]} '{p.strip()}' and '{e.strip()}' {prompt_all[1]}\n"
            best_file.write(best_prompt)
            all_file.write(all_prompt)

        pyt_file.close(); easy_file.close()
        best_file.close(); all_file.close()



if __name__ == '__main__':
    m = Master()
    """m.evaluate_ocr_outputs("NZZ_CORPUS/pytesseract_output_1780_1946", "NZZ_CORPUS/easyocr_output_1780_1946",
                           "NZZ_CORPUS/sentences_1780_1946", word_level=False)"""
    m.evaluate_llm_outputs("NZZ_CORPUS/best_chatgpt3", "NZZ_CORPUS/all_chatgpt3",
                           "NZZ_CORPUS/sentences_1780_1946",
                           "NZZ_CORPUS/pytesseract_output_1780_1946", word_level=True)
    """m.compare_two_llm_bests("CLEAR_CORPUS/llm_best", "CLEAR_CORPUS/chatgpt3_best",
                            "CLEAR_CORPUS/sentences", "CLEAR_CORPUS/pytesseract_output", word_level=False)"""
    # m.get_ocr_outputs_for_NZZ_corpus((1780, 1946), 200)
    # m.make_prompts_file()

