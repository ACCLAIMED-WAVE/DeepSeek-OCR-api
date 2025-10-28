import os
import fitz
import img2pdf
import io
import re
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE, GPU_MEMORY_UTILIZATION

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr import DeepseekOCRForCausalLM

from vllm.model_executor.models.registry import ModelRegistry

from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True, 
    max_model_len=8192,
    swap_space=0,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    disable_mm_preprocessor_cache=True
)

logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids= {128821, 128822})] #window for fast；whitelist_token_ids: <td>,</td>

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
    include_stop_str_in_output=True,
)


class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m' 

def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """
    pdf2images
    """
    images = []
    
    pdf_document = fitz.open(pdf_path)
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
        
        images.append(img)
    
    pdf_document.close()
    return images

def pil_to_pdf_img2pdf(pil_images, output_path):

    if not pil_images:
        return
    
    image_bytes_list = []
    
    for img in pil_images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)
    
    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    except Exception as e:
        print(f"error: {e}")



def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)


    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):


    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, jdx, output_path):

    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20, )
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{output_path}/images/{jdx}_{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1
                        
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)
                            
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                                    fill=(255, 255, 255, 30))
                        
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, jdx, output_path):
    result_image = draw_bounding_boxes(image, ref_texts, jdx, output_path)
    return result_image


def process_single_image(image, prompt_in):
    """single image"""
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {"image": DeepseekOCRProcessor().tokenize_with_images(images = [image], bos=True, eos=True, cropping=CROP_MODE)},
    }
    return cache_item


def run_ocr_on_pdf(llm, sampling_params, input_pdf_path: str, output_dir: str):
    """
    Processes a PDF file to extract text, layout, and images.

    Args:
        llm: The initialized VLLM model.
        sampling_params: The sampling parameters for generation.
        input_pdf_path: Path to the input PDF file.
        output_dir: Directory to save the output files.

    Returns:
        A tuple containing:
        - The final markdown content.
        - A list of paths to the extracted images.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/images', exist_ok=True)
    
    logger.debug(f'{Colors.RED}PDF loading .....{Colors.RESET}')
    images = pdf_to_images_high_quality(input_pdf_path)
    logger.debug(f'{Colors.RED}PDF loaded {len(images)} images{Colors.RESET}')


    prompt = PROMPT

    # batch_inputs = []
    logger.debug(f'{Colors.RED}Processing images with {NUM_WORKERS} workers{Colors.RESET}')
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:  
        batch_inputs = list(tqdm(
            executor.map(lambda image: process_single_image(image, prompt), images),
            total=len(images),
            desc="Pre-processed images"
        ))
    logger.debug(f'{Colors.RED}Images pre-processed{Colors.RESET}')


    # for image in tqdm(images):

    #     prompt_in = prompt
    #     cache_list = [
    #         {
    #             "prompt": prompt_in,
    #             "multi_modal_data": {"image": DeepseekOCRProcessor().tokenize_with_images(images = [image], bos=True, eos=True, cropping=CROP_MODE)},
    #         }
    #     ]
    #     batch_inputs.extend(cache_list)

    logger.debug(f'{Colors.RED}Generating outputs{Colors.RESET}')
    outputs_list = llm.generate(
        batch_inputs,
        sampling_params=sampling_params
    )
    logger.debug(f'{Colors.RED}Outputs generated{Colors.RESET}')

    mmd_det_path = os.path.join(output_dir, os.path.basename(input_pdf_path).replace('.pdf', '_det.mmd'))
    mmd_path = os.path.join(output_dir, os.path.basename(input_pdf_path).replace('.pdf', '.mmd'))
    pdf_out_path = os.path.join(output_dir, os.path.basename(input_pdf_path).replace('.pdf', '_layouts.pdf'))
    
    contents_det = ''
    contents = ''
    draw_images = []
    all_image_paths = []
    jdx = 0
    logger.debug(f'{Colors.RED}Combining outputs{Colors.RESET}')
    for output, img in zip(outputs_list, images):
        logger.debug(f'{Colors.BLUE}Processing output {jdx}{Colors.RESET}')
        content = output.outputs[0].text

        if '<｜end▁of▁sentence｜>' in content: # repeat no eos
            logger.debug(f'{Colors.BLUE}Removing EOS token{Colors.RESET}')
            content = content.replace('<｜end▁of▁sentence｜>', '')
        else:
            if SKIP_REPEAT:
                logger.debug(f'{Colors.BLUE}Skipping repeated content{Colors.RESET}')
                continue

        logger.debug(f'{Colors.BLUE}Adding page split{Colors.RESET}')
        page_num = f'\n<--- Page Split --->'

        contents_det += content + f'\n{page_num}\n'

        image_draw = img.copy()

        logger.debug(f'{Colors.BLUE}Processing references and matches{Colors.RESET}')
        matches_ref, matches_images, mathes_other = re_match(content)
        # print(matches_ref)
        result_image = process_image_with_refs(image_draw, matches_ref, jdx, output_dir)

        draw_images.append(result_image)

        logger.debug(f'{Colors.BLUE}Processing {len(matches_images)} image matches{Colors.RESET}')
        for idx, a_match_image in enumerate(matches_images):
            image_filename = f'{jdx}_{idx}.jpg'
            image_path = os.path.join(output_dir, 'images', image_filename)
            all_image_paths.append(image_path)
            content = content.replace(a_match_image, f'![](images/{image_filename})\n')

        logger.debug(f'{Colors.BLUE}Processing {len(mathes_other)} other matches{Colors.RESET}')
        for idx, a_match_other in enumerate(mathes_other):
            content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

        contents += content + f'\n{page_num}\n'

        jdx += 1

    logger.debug(f'{Colors.RED}Writing outputs{Colors.RESET}')
    with open(mmd_det_path, 'w', encoding='utf-8') as afile:
        afile.write(contents_det)

    with open(mmd_path, 'w', encoding='utf-8') as afile:
        afile.write(contents)

    logger.debug(f'{Colors.RED}Converting images to PDF{Colors.RESET}')
    pil_to_pdf_img2pdf(draw_images, pdf_out_path)
    logger.debug(f'{Colors.RED}PDF converted{Colors.RESET}')
    
    return contents, all_image_paths


if __name__ == "__main__":

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f'{OUTPUT_PATH}/images', exist_ok=True)
    
    run_ocr_on_pdf(
        llm=llm,
        sampling_params=sampling_params,
        input_pdf_path=INPUT_PATH,
        output_dir=OUTPUT_PATH
    )

