"""
Streamlit demo for MATHGPT Chain-of-Thought reasoning.
Alternative interface for Hugging Face Spaces.
"""

import streamlit as st
import requests
import json
import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

st.set_page_config(
    page_title="MATHGPT - Chain-of-Thought Reasoning",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 MATHGPT: Chain-of-Thought Mathematical Reasoning")
st.markdown("**GPT-2 fine-tuned for step-by-step mathematical problem solving**")

# Sidebar
st.sidebar.header("Settings")
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 200)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "This demo showcases a GPT-2 model fine-tuned on GSM8K "
    "mathematical reasoning dataset with Chain-of-Thought prompting."
)

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Ask a Math Question")
    
    # Sample questions
    sample_questions = [
        "Sarah has 5 apples and buys 3 more. How many apples does she have?",
        "A train travels 120 miles in 2 hours. What is its speed?",
        "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. How many eggs does she sell?",
        "If I save $25 per week for 8 weeks, how much will I have saved?",
        "A rectangle has length 12 cm and width 8 cm. What is its area?"
    ]
    
    selected_sample = st.selectbox("Or choose a sample question:", [""] + sample_questions)
    
    # Text input
    if selected_sample:
        question = st.text_area("Your math question:", value=selected_sample, height=100)
    else:
        question = st.text_area("Your math question:", height=100, 
                               placeholder="Enter your mathematical word problem here...")
    
    # Solve button
    if st.button("🚀 Solve Problem", type="primary"):
        if question.strip():
            with st.spinner("🤔 Thinking step by step..."):
                try:
                    # Try to call local API first, fallback to direct inference
                    try:
                        # Call API
                        response = requests.post(
                            "http://localhost:8000/solve",
                            json={"question": question, "max_tokens": max_tokens},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            reasoning = result["reasoning"]
                            answer = result["final_answer"]
                            confidence = result["confidence"]
                        else:
                            st.error(f"API Error: {response.status_code}")
                            reasoning = "Error calling API"
                            answer = None
                            confidence = 0.0
                            
                    except:
                        # Fallback to direct inference
                        st.info("API not available, using direct inference...")
                        
                        # Import and use inference directly
                        try:
                            sys.path.append(str(project_root / "api"))
                            from inference_service import MathSolver
                            
                            solver = MathSolver()
                            result = solver.solve(question, max_tokens)
                            reasoning = result["reasoning"]
                            answer = result["answer"]
                            confidence = result["confidence"]
                            
                        except Exception as e:
                            st.error(f"Inference error: {str(e)}")
                            reasoning = "Unable to solve problem"
                            answer = None
                            confidence = 0.0
                    
                    # Store results in session state
                    st.session_state.last_question = question
                    st.session_state.last_reasoning = reasoning
                    st.session_state.last_answer = answer
                    st.session_state.last_confidence = confidence
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question first!")

with col2:
    st.header("🧠 Step-by-Step Solution")
    
    # Display results
    if hasattr(st.session_state, 'last_reasoning'):
        st.markdown("### 💭 Reasoning Process:")
        st.text_area("", value=st.session_state.last_reasoning, height=300, disabled=True)
        
        st.markdown("### 🎯 Final Answer:")
        if st.session_state.last_answer:
            st.success(f"**{st.session_state.last_answer}**")
        else:
            st.warning("Could not extract numerical answer")
        
        st.markdown(f"**Confidence:** {st.session_state.last_confidence:.1%}")
        
        # Download solution
        if st.button("📥 Download Solution"):
            solution_text = f"""
Question: {st.session_state.last_question}

Reasoning:
{st.session_state.last_reasoning}

Final Answer: {st.session_state.last_answer or 'N/A'}
Confidence: {st.session_state.last_confidence:.1%}
            """
            st.download_button(
                "Download as Text",
                solution_text,
                file_name="mathgpt_solution.txt",
                mime="text/plain"
            )
    else:
        st.info("👆 Enter a math question on the left to see the step-by-step solution here!")
        
        # Show example
        st.markdown("### Example Output:")
        st.code("""
Question: Sarah has 5 apples and buys 3 more. How many apples does she have?

Let me solve this step by step:
Sarah starts with 5 apples.
She buys 3 more apples.
To find the total, I need to add: 5 + 3 = 8
Therefore, the answer is 8.

Final Answer: 8
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using GPT-2, PyTorch, and Streamlit | "
    "[GitHub](https://github.com/routsourav1729/MATHGPT) | "
    "**MATHGPT v1.0**"
)