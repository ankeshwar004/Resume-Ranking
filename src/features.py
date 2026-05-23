def skill_overlap(resume_texts, jd_texts, all_skills):
    rows = []

    for res, jd in zip(resume_texts, jd_texts):
        res_tokens = set(res.split())
        jd_tokens = set(jd.split())

        res_skills = res_tokens & all_skills
        jd_skills = jd_tokens & all_skills

        matched = res_skills & jd_skills

        match_count = len(matched)
        jd_count = len(jd_skills)
        res_count = len(res_skills)

        jd_coverage = match_count / jd_count if jd_count > 0 else 0

        # ---- CAPPED Precision (no penalty for extra useful skills) ----
        precision_capped = match_count / min(res_count, jd_count) if min(res_count, jd_count) > 0 else 0


        rows.append([
            match_count,
            # jd_coverage,
            precision_capped,

        ])

    return np.array(rows)

