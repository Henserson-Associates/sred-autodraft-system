from __future__ import annotations

import textwrap

import streamlit as st

from agents import ReportGenerator


st.set_page_config(page_title="SR&ED Report Generator", layout="wide")

st.title("SR&ED Report Generator")
st.write(
    "Provide your project details below. The system will retrieve similar approved SR&ED examples "
    "and assemble draft sections for you to refine."
)

with st.sidebar:
    st.header("Project Inputs")
    industry = st.text_input("Industry", placeholder="e.g., pharmacy")
    tech_code = st.text_input("Tech Code", placeholder="e.g., 01.01")
    project_description = st.text_area(
        "Project Description",
        height=160,
        placeholder="Briefly describe the SR&ED project you are claiming.",
    )

    generate = st.button("Generate Draft")

output_placeholder = st.empty()

if generate:
    if not project_description.strip():
        st.warning("Please provide a project description before generating a draft.")
    else:
        try:
            generator = ReportGenerator()
            sections = generator.generate_report(
                industry=industry.strip(),
                tech_code=tech_code.strip(),
                project_description=project_description.strip(),
            )

            with output_placeholder.container():
                st.subheader("Draft SR&ED Report Sections")
                for key, content in sections.items():
                    st.markdown(f"### {key.replace('_', ' ').title()}")
                    st.markdown(textwrap.dedent(content))
        except Exception as exc:
            st.error(f"Error generating report: {exc}")

