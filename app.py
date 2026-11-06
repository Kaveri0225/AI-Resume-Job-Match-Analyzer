import streamlit as st
import re
import sys
import subprocess

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "pypdf==5.9.0"
])

from pypdf import PdfReader


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume & Job Match Analyzer")

st.write(
    "Analyze your resume against a job description "
    "and identify your matching and missing skills."
)


skills_list = [
    "Python",
    "Java",
    "C++",
    "SQL",
    "Machine Learning",
    "Deep Learning",
    "Data Structures",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "HTML",
    "CSS",
    "JavaScript",
    "React",
    "Flask",
    "Django",
    "Streamlit",
    "Git",
    "GitHub",
    "Artificial Intelligence",
    "Data Analysis",
    "Excel",
    "Power BI",
    "REST API",
    "Communication",
    "Problem Solving",
    "OOP",
    "Database Management"
]



def extract_skills(text):

    extracted_skills = []

    for skill in skills_list:

        pattern = (
            r"(?<!\w)"
            + re.escape(skill)
            + r"(?!\w)"
        )

        if re.search(pattern, text, re.IGNORECASE):

            extracted_skills.append(skill)

    return extracted_skills



def extract_resume_text(pdf_file):

    reader = PdfReader(pdf_file)

    resume_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:

            resume_text += text + "\n"

    return resume_text




def calculate_similarity(resume_text, job_description):

    documents = [
        resume_text,
        job_description
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return similarity * 100



resume = st.file_uploader(
    "Upload Your Resume (PDF)",
    type=["pdf"]
)




job_description = st.text_area(
    "Paste the Job Description Here",
    height=200,
    placeholder="Enter the job requirements here..."
)



if st.button("Analyze Resume", type="primary"):

    if resume is None:

        st.warning("Please upload your resume first.")

    elif not job_description.strip():

        st.warning("Please enter a job description.")

    else:

        try:

            # Extract resume text
            resume_text = extract_resume_text(resume)

            if not resume_text.strip():

                st.error(
                    "No text found in the PDF. "
                    "This may be a scanned resume."
                )

            else:

                # Extract resume skills
                extracted_skills = extract_skills(
                    resume_text
                )

                # Extract required skills
                required_skills = extract_skills(
                    job_description
                )


                matched_skills = [
                    item for item in required_skills
                    if item in extracted_skills
                ]


                missing_skills = [
                    item for item in required_skills
                    if item not in extracted_skills
                ]


                if len(required_skills) > 0:

                    match_score = (
                        len(matched_skills)
                        / len(required_skills)
                    ) * 100

                else:

                    match_score = 0


                try:

                    similarity = calculate_similarity(
                        resume_text,
                        job_description
                    )

                except ValueError:

                    similarity = 0


                st.header("📃 Resume Analysis")

                with st.expander(
                    "View Extracted Resume Text"
                ):

                    st.text(resume_text)


                st.subheader(
                    "🛠️ Skills Identified in Resume"
                )

                if extracted_skills:

                    st.write(
                        ", ".join(extracted_skills)
                    )

                else:

                    st.warning(
                        "No skills identified in resume."
                    )

                # ==================================
                # DISPLAY SCORES
                # ==================================

                st.subheader(
                    "📊 Job Match Results"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Skill Match Score",
                        f"{match_score:.2f}%"
                    )

                    st.progress(
                        int(match_score)
                    )

                with col2:

                    st.metric(
                        "TF-IDF Text Similarity",
                        f"{similarity:.2f}%"
                    )


                st.subheader(
                    "✅ Matching Skills"
                )

                if matched_skills:

                    for item in matched_skills:

                        st.write("✅", item)

                else:

                    st.write(
                        "No matching skills found."
                    )

                st.subheader(
                    "❌ Missing Skills"
                )

                if missing_skills:

                    for item in missing_skills:

                        st.write("❌", item)

                else:

                    st.success(
                        "All identified required skills are present!"
                    )


                st.subheader(
                    "💡 Improvement Suggestions"
                )

                if missing_skills:

                    st.write(
                        "Consider learning these skills "
                        "to improve your alignment with "
                        "the job description:"
                    )

                    st.write(
                        ", ".join(missing_skills)
                    )

                else:

                    st.success(
                        "Your resume contains all the "
                        "identified required skills."
                    )



                st.info(
                    "These scores are based on text and "
                    "skill matching. They are not official "
                    "ATS scores or guarantees of selection."
                )

        except Exception as e:

            st.error(
                f"An error occurred: {e}"
            ) 