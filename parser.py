import os
import re
import signal
import pytesseract
from PIL import Image
from docx import Document
from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr, constr, confloat, ValidationError
from email.utils import parseaddr
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader

class PersonalInformation(BaseModel):
    firstNameEN: str
    lastNameEN: str
    firstNameTH: str
    lastNameTH: str
    birthDate: Optional[str]
    age: Optional[int]
    gender: Optional[str] = Field(default=None, pattern="^(Male|Female|Other)$")
    phone: constr(min_length=10, max_length=15)
    email: EmailStr
    province: Optional[str]
    district: Optional[str]

class Salary(BaseModel):
    lastedSalary: Optional[confloat(ge=0)]
    expectSalary: Optional[confloat(ge=0)]

class Qualification(BaseModel):
    industry: Optional[str]
    experiencesYear: Optional[int]
    majorSkill: Optional[str]
    minorSkill: Optional[str]

class Certificate(BaseModel):
    course: Optional[str]
    year: Optional[str]
    institute: Optional[str]

class Experience(BaseModel):
    company: Optional[str]
    position: Optional[str]
    project: Optional[str]
    startDate: Optional[str]
    endDate: Optional[str]
    responsibility: Optional[str]

class Education(BaseModel):
    degreeLevel: str
    program: str
    major: str
    year: str
    university: str

class Resume(BaseModel):
    personalInformation: Optional[PersonalInformation]
    availability: Optional[str]
    currentPosition: Optional[str]
    salary: Optional[Salary]
    qualification: Optional[List[Qualification]]
    softSkills: Optional[List[str]]
    technicalSkills: Optional[List[str]]
    experiences: Optional[List[Experience]]
    educations: Optional[List[Education]]
    certificates: Optional[List[Certificate]]


resume_template = """
You are an AI assistant tasked with extracting structured information from a technical resume.
Only extract the information that is present in the Resume class.
The resume may be written in either Thai or English.

IMPORTANT:
- Do not translate any part of the input text.
- Preserve all content exactly in the language it appears (Thai or English).
- Extract values into the Resume class **using the original language of each field**.
- Map fields like firstNameTH and lastNameTH only if the names are written in Thai.
- Map fields like firstNameEN and lastNameEN only if the names are written in English.
- Do not guess or generate missing fields.

Resume Detail:
{resume_text}
"""

parser = PydanticOutputParser(pydantic_object=Resume)

prompt_template = PromptTemplate(
    template=resume_template,
    input_variables=['resume_text']
)

model = init_chat_model(model='gpt-4o-mini', model_provider='openai').with_structured_output(Resume)


class TimeoutException(Exception): pass

def timeout_handler(signum, frame): raise TimeoutException()

def extract_tesseract_text(file_path: str, timeout=600) -> str:
    print(f"[Tesseract] Running Tesseract on image: {file_path}")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        image = Image.open(file_path)
        custom_config = "--oem 3 --psm 4 -l tha+eng"
        text = pytesseract.image_to_string(image, config=custom_config)

        if len(text.strip()) < 100:
            print("[Tesseract] Fallback to PSM 6 (text too short)")
            fallback_config = "--oem 3 --psm 6 -l tha+eng"
            text = pytesseract.image_to_string(image, config=fallback_config)

        return text.strip()

    except TimeoutException:
        print("[Tesseract] OCR timed out.")
        return "OCR timed out. Try a smaller or clearer image."
    except Exception as e:
        print(f"[Tesseract] OCR failed: {repr(e)}")
        return f"OCR failed: {str(e)}"
    finally:
        signal.alarm(0)

def smart_resize_image(path: str, max_width: int = 1000, max_height: int = 1000):
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        width, height = img.size

        if height > width:
            if height <= max_height:
                print(f"[Resize] No resizing needed (height={height} <= {max_height})")
                return
            new_height = max_height
            scale_factor = new_height / height
            new_width = int(width * scale_factor)
        else:
            if width <= max_width:
                print(f"[Resize] No resizing needed (width={width} <= {max_width})")
                return
            new_width = max_width
            scale_factor = new_width / width
            new_height = int(height * scale_factor)

        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        resized_img.save(path, quality=80, optimize=True)
        print(f"[Resize] Resized to {new_width}x{new_height}")
    except Exception as e:
        print(f"[Resize] Failed to resize image: {e}")


def extract_text_from_image(file_path: str) -> str:
    print(f"[Image] Start processing image: {file_path}")
    smart_resize_image(file_path, max_width=1000, max_height=1000)

    print(f"[Image] Running Tesseract OCR pipeline...")
    text = extract_tesseract_text(file_path)
    print(f"[Image] Finished Tesseract OCR. Extracted text length: {len(text)}")

    return text


def extract_text_from_pdf(file_path: str) -> str:
    print(f"[PDF] Start loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"[PDF] Loaded {len(docs)} pages.")

    text = "\n".join([doc.page_content for doc in docs])
    print(f"[PDF] Extracted text length: {len(text)}")

    return text


def extract_text_from_docx(file_path: str) -> str:
    print(f"[DOCX] Start loading DOCX file: {file_path}")
    doc = Document(file_path)

    paragraphs = [para.text for para in doc.paragraphs]
    print(f"[DOCX] Found {len(paragraphs)} paragraphs.")

    text = "\n".join(paragraphs)
    print(f"[DOCX] Extracted text length: {len(text)}")

    return text


def clean_invalid_emails(text: str) -> str:
    pattern = r'\b[\w\.-]+@[\w\.-]+\.\w+\b'
    matches = re.findall(pattern, text)
    for match in matches:
        if not re.fullmatch(r'[\w\.-]+@[\w\.-]+\.\w{2,}', match):
            print(f"[Warning] Found possibly invalid email: {match}")
            text = text.replace(match, "")
    return text

def normalize_thai_phone_number(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    if digits.startswith("66"):
        digits = "0" + digits[2:]
    if digits.startswith("0") and len(digits) == 10:
        return digits
    return digits

def load_text_set(filepath: str) -> set:
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

valid_provinces = load_text_set("dataset-provinces.txt")
valid_technical_skills = load_text_set("dataset-technical-list.txt")

def validate_province(province: Optional[str], valid_provinces: set) -> str:
    if not province:
        print("[Location] Missing province")
        return ""
    if province not in valid_provinces:
        print(f"[Location] Province not in list: {province}")
        return province
    return province

def validate_technical_skills(skills: Optional[List[str]], valid_skills: set) -> Optional[List[str]]:
    if not skills:
        return skills
    validated = []
    for skill in skills:
        if skill in valid_skills:
            validated.append(skill)
        else:
            print(f"[Technical Skill] Skill not in list: {skill} (keeping original)")
            validated.append(skill)
    return validated

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def parse_resume(file_path: str) -> Resume:
    print(f"[Parse] Start parsing: {file_path}")
    resume_text = extract_text(file_path)
    resume_text = clean_invalid_emails(resume_text)
    prompt = prompt_template.invoke({"resume_text": resume_text})
    result = model.invoke(prompt)

    if result.personalInformation:
        if result.personalInformation.phone:
            original = result.personalInformation.phone
            formatted = normalize_thai_phone_number(original)
            print(f"[Phone] Normalized phone number: {original} → {formatted}")
            result.personalInformation.phone = formatted

        result.personalInformation.province = validate_province(
            result.personalInformation.province, valid_provinces
        )

    if result.technicalSkills:
        result.technicalSkills = validate_technical_skills(
            result.technicalSkills, valid_technical_skills
        )
    
    print("[Parse] Resume parsing complete.")
    return result, resume_text