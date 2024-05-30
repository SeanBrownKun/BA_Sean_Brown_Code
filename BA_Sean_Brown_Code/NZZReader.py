import xml.etree.ElementTree as ET
import os


class NZZReader:
    """This static class deals with reading the NZZ corpus and returning:
    (filename, coordinate (within img file), language, text (within coordinates))
    for each section in each file in the corpus, where one file is an image, and
    one section is a paragraph in the image."""

    def __init__(self):
        pass

    @staticmethod
    def readall(range_tuple=(1780, 1946)):
        """This method goes through each file and yields all entries,
        and collects them in a list that is returned to the Master class
        for generating NZZ corpus OCR predictions and prompt files."""
        xml_directory = "../NZZ-black-letter-ground-truth-master/xml/NZZ_groundtruth"

        all_entries = []

        idx = 0
        for filename in os.listdir(xml_directory):
            idx += 1
            if idx % 10 == 0:
                print(f"reading file of year {filename[4:8]} / 1946")
            filepath = os.path.join(xml_directory, filename)
            if os.path.isfile(filepath):
                entries_in_file, prob = NZZReader.read_one_file(filepath, range_tuple)
                if not prob:
                    all_entries.extend(entries_in_file)
        print("NZZ read!")
        return all_entries


    @staticmethod
    def read_one_file(filepath, range_tuple):
        """This method reads all paragraphs in one file."""
        entries = []  # entries have format: [img_filename, coords_tuple, text] for each entry.
        after, before = range_tuple

        tree = ET.parse(filepath)
        root = tree.getroot()
        for child in root:
            if child.tag.endswith("Page"):
                equivalent_img_filename = child.attrib["imageFilename"]
                # check if it's within year range
                if equivalent_img_filename.startswith("nzz"):
                    year = equivalent_img_filename[4:8]
                else:
                    year = equivalent_img_filename[9:13]
                if int(year) < after or int(year) > before:
                    continue

                for sub_child in child:
                    if sub_child.tag.endswith("TextRegion"):
                        coords = None
                        language = None
                        for sub_text in sub_child:
                            # get language
                            if sub_text.tag.endswith("TextLine"):
                                if "primaryLanguage" in sub_text.attrib:
                                    language = sub_text.attrib["primaryLanguage"]

                            # extract coords in correct format
                            if sub_text.tag.endswith("Coords"):
                                c = sub_text.attrib["points"].split()
                                if len(c) < 3:
                                    return None, True
                                mins = c[0].split(",")
                                maxs = c[2].split(",")
                                if int(mins[0]) >= int(maxs[0]) or int(mins[1]) >= int(maxs[1]):
                                    return None, True
                                coords = (int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1]))

                            # get text
                            elif sub_text.tag.endswith("TextEquiv"):
                                for tep in sub_text:
                                    assert coords
                                    language = language if language else "German"
                                    entries.append((equivalent_img_filename, coords, language, tep.text))
        return entries, False

    @staticmethod
    def load_predictions(ocr):
        """This method was replaced by the Master class.
        I keep it as a form of version control."""
        original = []
        predictions = []

        entries = NZZReader.readall()
        for entry in entries:
            img_file = f"../NZZ-black-letter-ground-truth-master/img/{entry[0]}"
            coordinates = entry[1]
            # assert entry[2] in languages
            # language = languages[entry[2]]
            ground_truth = entry[3].strip()
            original.append(ground_truth)

            predictions = ocr.prediction_from_all_ocr_models(img_file, crop_tuple=coordinates)
            predictions.append(predictions)

        original = [" ".join(x) for x in original]
        predictions = [" ".join(x) for x in predictions]
        return original, predictions
