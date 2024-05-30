from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from time import perf_counter


class LLM:
    """This class contains the Fastchat-t5-3b-v1.0 LLM
     and generates predictions from it."""

    def __init__(self):
        a = perf_counter()
        self.tokenizer = T5Tokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")

        self.best_result_prompt = [
            ("My OCR system gave me this output:", "Please correct it. It's very important that you only correct the output. Do not make further comments!"),
            ("correct this phrase: ", "")
        ]

        self.all_result_prompt = [
            ("My two OCR systems gave me these outputs:", "Please guess the original input. Only give me the one phrase. Do not make further comments!"),
            ("these phrases are all wrong:", "Give me your prediction of the phrase I'm looking for.")
        ]
        b = perf_counter()
        print(f"LLM loaded in {b - a} seconds.")

    def append_to_prompts(self, prompt_file, prompt):
        """This method adds a prompt to the prompt file."""
        with open(prompt_file, "a", encoding='utf-8') as file:
            file.write(f"{prompt}\n")

    def __assemble_prompt(self, result, prompt):
        """This method generates a LLM-readable prompt."""
        if type(result) == list:
            interjection = "' and '"
            p_all = f"'{prompt[0]} '{interjection.join(result)}' {prompt[1]}'"
            return f"### Human: {p_all}\n### Assistant: "
        else:
            p_best = f"'{prompt[0]} '{result}' {prompt[1]}'"
            return f"### Human: {p_best}\n### Assistant: "

    def predict_prompt(self, result, prompt, needs_assembly=True):
        """This method feeds the prompt to the LLM
        and generates a prediction."""
        before = perf_counter()
        if needs_assembly:
            prompt = self.__assemble_prompt(result, prompt)
        print(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
        )
        after = perf_counter()
        print(f"LLM predicted in: {after - before} seconds.")
        return self.tokenizer.decode(tokens[0])

    def predict_all(self, result):
        """This method is the wrapper method that goes through the
        entire prompt file and makes predictions"""
        best_result = result[0]

        best_result_indexes = {}
        result_indexes = {}

        index = 0
        for prompt in self.best_result_prompt:
            best_result_indexes[index] = self.predict_prompt(best_result, prompt); index += 1

        index = 0
        for prompt in self.all_result_prompt:
            result_indexes[index] = self.predict_prompt(result, prompt); index += 1

        return best_result_indexes, result_indexes

    def generate_from_file_prompts(self, filename, outname):
        """This method saves the predictions to the LLM predictions file."""
        all_outs = []
        with open(filename, "r", encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                prompt = f"### Human: {line}\n### Assistant: "
                out = self.predict_prompt(None, prompt, needs_assembly=False)
                all_outs.append(out)
        with open(outname, "w", encoding='utf-8') as file:
            for out in all_outs:
                file.write(f"{out}\n")


if __name__ == '__main__':
    llm = LLM()
    # llm.generate_from_file_prompts("NZZ_CORPUS/all_prompts", "NZZ_CORPUS/llm_all_NZZ")
    # llm.generate_from_file_prompts("NZZ_CORPUS/best_prompts", "NZZ_CORPUS/llm_best_NZZ")
