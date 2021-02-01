import os
import json
import argparse

from xml.etree.ElementTree import Element, SubElement, ElementTree

PREVIEW_ROOT = "/mnt/raid/davech2y/ScanNet_previews/"

def create_text_cell(text):
    td = Element("td")
    td.text = text

    return td

def create_image_cell(image_path):
    td = Element("td")
    img = SubElement(td, attrib={"src": image_path, "width": "300px"})

    return td

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, default="./predictions.json")
    parser.add_argument('--image_dir', type=str)
    args = parser.parse_args()

    predictions = json.load(open(args.predictions))
    scan_list = sorted(list(predictions.keys()))

    doc = Element("html")
    table = SubElement(doc, "table")
    for scan_id in scan_list:
        image_list = sorted([int(k) for k in predictions[scan_id].keys()]) # int

        text_list = [scan_id]
        text_list += ["{}-{}".format(scan_id, str(image_id)) for image_id in image_list]
        
        des_list = [""]
        des_list += [predictions[scan_id][str(image_id)] for image_id in image_list]

        image_list = [os.path.join(PREVIEW_ROOT, "{}_vh_clean_2.png".format(scan_id))]
        image_list += [os.path.join(args.image_dir, scan_id, "color", "{}.jpg").format(str(image_id)) for image_id in image_list]
        
        num_iters = len(image_list)
        
        for r in range(3):
            tr = SubElement(table, "tr")
            for i in range(num_iters):
                text = text_list[i]
                image = image_list[i]
                des = des_list[i]

                if r == 0: # text
                    td = create_text_cell(text)
                    tr.append(td)
                elif r == 1: # image
                    td = create_image_cell(image)
                    tr.append(td)
                else: # des
                    td = create_text_cell(text)
                    tr.append(td)

        break

    with open("predictions.html", "w") as f:
        doc.write(f)