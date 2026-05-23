import re


def extract_experience(text,is_resume=True):
  text=text.lower()

  matches_range=re.findall(r'(\d+)\s*-\s*(\d+)\s*(years|year|yrs|yr)',text)
  if matches_range:
    values=[(int(a)+int(b))/2 for a,b,_ in matches_range]
    return max(values)

  matches = re.findall(r'(\d+)\s*\+?\s*(years|year|yrs|yr)',text)
  if matches:
    values= [int(match[0]) for match in matches]
    return sum(values) if is_resume else max(values)

  return 0




def normalize_skills(skill):
    skill = str(skill).lower().strip()
    # Replace separators with space
    skill = re.sub(r'[/,_\-]+', ' ', skill)
    # Remove weird patterns like e.g.
    skill = re.sub(r'\be\.g\.\b', '', skill)
    # Remove extra spaces
    skill = re.sub(r'\s+', ' ', skill)

    return skill.strip()

def split_skills(skill):
    return re.split(r'\bor\b|/|,|and|___|__', skill)

def extract_skills(skill):
    raw = str(skill).lower()
    parts = split_skills(raw)

    cleaned = []
    for p in parts:
        p = normalize_skills(p)
        if p:
            cleaned.append(p)

    return cleaned

def is_valid_skills(skill):
    if len(skill) < 2:
        return False
    if len(skill) > 40:
        return False
    if skill.isdigit():
        return False
    return True


def extract_skills_from_row(df,col):
  skills_set = set()
  for skills in df[col].dropna():
      extracted = extract_skills(skills)
      for s in extracted:
          if is_valid_skills(s):
              skills_set.add(s)
  return skills_set
