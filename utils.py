import os
import fitz

def extract_blocks(pdf_path):
    blocks = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        for block in page.get_text("blocks"):
            blocks.append({
                'page': page_num,
                'x0': block[0],
                'y0': block[1],
                'x1': block[2],
                'y1': block[3],
                'raw_block': block
            })
    doc.close()
    return blocks

def drop_to_file(block_text, block_type, block_page_number):
    with open("output.txt", "a", encoding='utf-8') as file:
        if block_type == 'Header':
            file.write(f"<h1>{block_text}</h1>\n<{block_page_number + 1}>\n\n")
        elif block_type == 'Body':
            file.write(f"<body>{block_text}</body>\n<{block_page_number + 1}>\n\n")
        elif block_type == 'Footer':
            file.write(f"<footer>{block_text}</footer>\n<{block_page_number + 1}>\n\n")
        elif block_type == 'Quote':
            file.write(f"<blockquote>{block_text}</blockquote>\n<{block_page_number + 1}>\n\n")
        else:
            file.write(f"{block_text} ERROR\n\n")

def delete_if_exists(del_file):
    if os.path.exists(del_file):
        os.remove(del_file)

