import os
import json

from dotenv import load_dotenv
from groq import Groq


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. Check your .env file."
    )


# ==========================================
# 2. INITIALIZE GROQ CLIENT
# ==========================================

client = Groq(api_key=api_key)


# ==========================================
# 3. GET THE COMPLETE JOB DESCRIPTION
# ==========================================

print("\n========== ZEURX JOB DESCRIPTION ANALYZER ==========\n")

print("Paste the complete job description below.")
print("When finished, type END on a new line and press Enter.\n")

lines = []

while True:
    line = input()

    if line.strip() == "END":
        break

    lines.append(line)

job_description = "\n".join(lines).strip()

if not job_description:
    raise ValueError("Job description cannot be empty.")

print("\nJob description received successfully.")
print(f"Characters received: {len(job_description)}")


# ==========================================
# 4. ANALYZE THE JOB DESCRIPTION
# ==========================================

print("\nAnalyzing the job description with AI...")

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "system",
            "content": """
You are ZeurX, an AI-powered job description analysis engine.

Your task is to extract structured information from job descriptions.

Return one valid JSON object with exactly these eight fields:

{

    "job_title": "string",
    "required_skills": [],
    "preferred_skills": [],
    "technical_skills": [],
    "experience_requirements": [],
    "responsibilities": [],
    "ai_technologies": [],
    "other_requirements": []

}

FIELD DEFINITIONS

1. job_title:
  Extract the job title explicitly stated in the job description.
  Look for titles in headings, opening sentences, and phrases describing
  the position being advertised.

  Examples of valid titles include "AI Engineer", "Junior AI Engineer",
  "Generative AI Developer", and "Backend Engineer".

  Return the title as a concise string, preserving the seniority level
  and specialization when stated.

  Do not use the company name, department name, or a list of eligible
  applicants as the job title.

  If no job title can be identified from the description, return
  "Not specified". Do not invent a title based only on the responsibilities.

2. technical_skills:
   Extract programming languages, frameworks, libraries, databases,
   APIs, developer tools, infrastructure technologies, and technical
   concepts required or mentioned in the description.

3. experience_requirements:
  Extract only explicitly stated requirements
  concerning a candidate's experience level, duration, or practical experience.

  Include:
  1. Explicit years or duration of experience, such as "3+ years of experience".
  2. Explicit experience levels, such as "entry-level" or "senior", when
     stated as a candidate requirement.
  3. Explicit practical or hands-on experience requirements, such as
     "Hands-on experience building LLM-based conversational AI systems".

  Exclude:
  1. General skill requirements, such as "Strong proficiency in Python".
  2. Knowledge or understanding requirements, such as "Solid understanding
     of RAG".
  3. Responsibilities that describe what the employee will do.
  4. Experience mentioned only in the company introduction.
  5. Experience explicitly listed under a preferred, desirable, optional,
     bonus, or nice-to-have section. Such experience belongs in
     preferred_skills, not experience_requirements.

  Do not invent years, experience levels, or qualifications.
  If no explicit experience requirement exists, return an empty list.

  Do not invent years of experience, qualifications, or experience levels.
  If the job description does not specify any experience-related requirement, return an empty list.
  Preserve the original meaning and wording as closely as possible.

4. responsibilities:
   Extract the duties, tasks, deliverables, and activities the employee
   is expected to perform.

5. ai_technologies:
   Extract AI-specific technologies, models, techniques, architectures,
   and methods, including LLMs, RAG, embeddings, AI agents, and related
   technologies when explicitly mentioned.

6. preferred_skills:
   Extract skills or qualifications explicitly described as preferred,
   desirable, optional, or a bonus.

7. other_requirements:
   Extract other explicit requirements, including location, work
   arrangement, communication skills, language requirements, and
   relevant nontechnical requirements.

EXTRACTION RULES

- Read the entire job description before producing the result.
- Extract every relevant item, not just the first few.
- Preserve the meaning and wording of the original description.
- Do not invent skills, requirements, responsibilities, or qualifications.
- Do not treat an unstated requirement as an explicit requirement.
- An item may appear in more than one category when it genuinely
  belongs to both categories.
- If a category has no relevant information, return an empty array.
- Every array must contain strings.
- Return valid JSON only, without Markdown fences or additional text.


REQUIRED SKILLS:
Extract skills and qualifications that the employer explicitly
states candidates must have or demonstrate.

Examples of requirement wording:
- "You must have..."
- "Required skills include..."
- "Strong fundamentals in..."
- "You should understand..."
- "Knowledge of ... is required."

PREFERRED SKILLS:
Extract skills explicitly described as preferred, desirable,
advantageous, optional, or a bonus.

Examples:
- "Experience with X is a plus."
- "Knowledge of X would be an advantage."
- "Preferred qualifications include..."

IMPORTANT:
- Do not classify every mentioned technology as required.
- Use the surrounding sentence to determine whether a skill
  is required, preferred, or simply mentioned.
- Do not invent requirements.
- Keep responsibilities separate from skills.
- A skill can appear in technical_skills and required_skills
  when it is both a technical skill and an explicit requirement.
- If no required or preferred skills are explicitly stated,
  return an empty array for that category.
  

CLASSIFICATION RULES

- technical_skills:
  Extract technical skills, tools, frameworks, programming languages,
  databases, and technical concepts mentioned in the description.
  Do not assume every mentioned skill is mandatory.

- experience_requirements:
  Extract explicit requirements concerning years of experience,
  experience levels, qualifications, and hands-on experience.
  Preserve whether a requirement is mandatory or preferred.

- responsibilities:
  Include only duties and activities the employee is expected to
  perform as part of the job.
  Do not include instructions about applying for the job.

- ai_technologies:
  Extract AI-specific technologies, models, techniques, and methods.
  Avoid duplicating an item unless it provides meaningful information
  in both categories.

- preferred_skills:
  Include qualifications explicitly described as preferred,
  desirable, optional, advantageous, or a bonus.
  Do not classify a skill as preferred merely because it is mentioned
  as relevant or because the description gives an example.

- other_requirements:
  Include location, work arrangement, communication requirements,
  and application instructions.Preserve whether each requirement is mandatory or preferred.
  If an educational qualification, certification, or other condition
  is explicitly described as preferred, desirable, optional, or
  nice-to-have, retain that status in the extracted text.
  Never present a preferred qualification as mandatory.
  Do not classify application instructions as job responsibilities.
  

IMPORTANT:
Distinguish mandatory requirements from preferred qualifications
based on the wording of the original job description.
Never invent a requirement or change an optional qualification
into a mandatory one.
"""
        },
        {
            "role": "user",
            "content": f"""
Analyze the following complete job description.

JOB DESCRIPTION START
{job_description}
JOB DESCRIPTION END

Extract the information according to the system instructions.

Return the complete JSON object with all eight required fields,
including required_skills and preferred_skills.
"""
        }
    ],
    temperature=0,
    response_format={"type": "json_object"},
)


# ==========================================
# 5. READ AND VALIDATE THE AI RESPONSE
# ==========================================

raw_response = response.choices[0].message.content

if not raw_response:
    raise ValueError("The AI returned an empty response.")

try:
    analysis = json.loads(raw_response)

except json.JSONDecodeError as error:
    raise ValueError(
        f"The AI returned invalid JSON: {error}"
    ) from error


required_fields = [
    "job_title",
    "required_skills",
    "preferred_skills",
    "technical_skills",
    "experience_requirements",
    "responsibilities",
    "ai_technologies",
    "other_requirements",
]

missing_fields = [
    field
    for field in required_fields
    if field not in analysis
]

if missing_fields:
    raise ValueError(
        f"The AI response is missing fields: {missing_fields}"
    )

for field in required_fields:
    if field == "job_title":
        if not isinstance(analysis[field], str):
            raise ValueError(
                "The job_title field must be a string."
            )
    else:
        if not isinstance(analysis[field], list):
            raise ValueError(
                f"The {field} field must be an array."
            )

        if not all(
            isinstance(item, str)
            for item in analysis[field]
        ):
            raise ValueError(
                f"Every item in {field} must be a string."
            )


# ==========================================
# 6. DISPLAY THE FINAL ANALYSIS
# ==========================================

print("\n" + "=" * 55)
print("ZEURX JOB ANALYSIS")
print("=" * 55)

print(json.dumps(analysis, indent=4, ensure_ascii=False))

print("\n" + "=" * 55)
print("Analysis completed.")
print("=" * 55)