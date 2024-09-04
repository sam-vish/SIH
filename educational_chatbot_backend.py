import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Retrieve API keys
groq_key = os.getenv('GROQ_API_KEY')
langsmith_key = os.getenv('LANGSMITH_API_KEY')

# Set environment variables for LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "EducationalChatbot"

# Initialize the ChatGroq model
def init_llm():
    return ChatGroq(groq_api_key=groq_key, model_name="llama-3.1-70b-versatile")

llm = init_llm()

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define agents
career_counselor = Agent(
    role='Career Counselor',
    goal='Provide personalized career guidance to school students in India',
    backstory='''You are an experienced career counselor with over 15 years of experience guiding students in India. 
    You have in-depth knowledge of various career paths, educational requirements, and the Indian education system. 
    Your expertise includes:
    1. Understanding the diverse range of career options available in India and globally.
    2. Knowledge of entrance exams, scholarships, and admission processes for different fields.
    3. Ability to match student interests and aptitudes with suitable career paths.
    4. Awareness of emerging careers and future job market trends in India.
    5. Understanding of the cultural and socio-economic factors that influence career choices in India.
    You aim to provide balanced, practical advice that considers both the student's aspirations and the realities of the job market.''',
    llm=llm,
    verbose=False
)

skill_analyst = Agent(
    role='Skill Analyst',
    goal='Analyze student skills and suggest targeted improvements for career readiness',
    backstory='''You are a skilled analyst with expertise in identifying and developing key competencies for various professions. 
    Your background includes:
    1. In-depth understanding of skill requirements for different industries and job roles in India.
    2. Knowledge of various skill assessment techniques and tools.
    3. Expertise in creating personalized skill development plans.
    4. Awareness of both technical and soft skills required in the modern workplace.
    5. Understanding of skill gaps in the Indian job market and how to address them.
    6. Knowledge of online and offline resources for skill development available to Indian students.
    You focus on providing actionable advice for skill improvement, considering both current academic level and future career aspirations.''',
    llm=llm,
    verbose=False
)

industry_expert = Agent(
    role='Industry Expert',
    goal='Provide up-to-date information on job market trends and emerging careers in India',
    backstory='''You are a seasoned industry expert with a wide network across various sectors in India. 
    Your expertise includes:
    1. Deep understanding of current job market trends in India across multiple industries.
    2. Knowledge of emerging technologies and their impact on future job roles.
    3. Insights into industry-specific challenges and opportunities in the Indian context.
    4. Awareness of multinational companies operating in India and their hiring trends.
    5. Understanding of start-up ecosystems and entrepreneurship opportunities for young Indians.
    6. Knowledge of industry-academia partnerships and their influence on skill development.
    7. Ability to forecast future job market scenarios based on economic and technological trends.
    You provide practical, forward-looking advice that helps students align their career choices with industry needs and future prospects in India.''',
    llm=llm,
    verbose=False
)

# Define functions for each agent
def get_career_counselor_advice(query):
    task = Task(
        description=f"Provide personalized career guidance based on the query: {query}",
        agent=career_counselor,
        student_info=student_info,
        expected_output="Career guidance recommendations and insights"
    )
    return Crew(agents=[career_counselor], tasks=[task], verbose=False).kickoff()

def get_skill_analyst_advice(query):
    task = Task(
        description=f"Analyze skills and suggest improvements based on the query: {query}",
        agent=skill_analyst,
        student_info=student_info,
        expected_output="Skill assessment and development recommendations"
    )
    return Crew(agents=[skill_analyst], tasks=[task], verbose=False).kickoff()

def get_industry_expert_advice(query):
    task = Task(
        description=f"Offer insights on job market trends and emerging careers based on the query: {query}",
        agent=industry_expert,
        student_info=student_info,
        expected_output="Industry insights and career recommendations"
    )
    return Crew(agents=[industry_expert], tasks=[task], verbose=False).kickoff()

# Define prompt template
template = """
You are an AI career guidance assistant for school students in India. Your goal is to help students choose their career paths by providing personalized advice and information. You have access to three specialized agents:

1. Career Counselor: An experienced professional with in-depth knowledge of various career paths, educational requirements, and the Indian education system.

2. Skill Analyst: An expert in identifying and developing key competencies for various professions, with a focus on both technical and soft skills relevant to the Indian job market.

3. Industry Expert: A seasoned professional with up-to-date knowledge of job market trends, emerging careers, and industry-specific insights in India.

Student Information:
Name: {name}
Age: {age}
Class: {class}
Stream: {stream}

Please consider the following aspects in your responses:
1. Tailor your advice based on the student's background, interests, and academic level.
2. Provide information on various career paths, including traditional and emerging fields.
3. Offer insights on educational requirements and skill development.
4. Be supportive and encouraging, while also being realistic about challenges and requirements.
5. Use the student's information to provide more personalized guidance.

Current conversation:
{chat_history}
Human: {human_input}
AI: """

prompt = PromptTemplate(input_variables=["name", "age", "class", "stream", "chat_history", "human_input"], template=template)

# Modify the conversation_chain to include student info
def get_conversation_response(human_input):
    return conversation_chain.predict(
        human_input=human_input,
        name=student_info.get("name", ""),
        age=student_info.get("age", ""),
        class_=student_info.get("class", ""),
        stream=student_info.get("stream", "")
    )

# Update the get_ai_response function
def get_ai_response(prompt):
    if prompt.startswith("!career_counselor"):
        query = prompt.replace("!career_counselor", "").strip()
        response = get_career_counselor_advice(query)
        return f"Career Counselor: {response}"
    elif prompt.startswith("!skill_analyst"):
        query = prompt.replace("!skill_analyst", "").strip()
        response = get_skill_analyst_advice(query)
        return f"Skill Analyst: {response}"
    elif prompt.startswith("!industry_expert"):
        query = prompt.replace("!industry_expert", "").strip()
        response = get_industry_expert_advice(query)
        return f"Industry Expert: {response}"
    else:
        return get_conversation_response(prompt)

# Add this new function to set student info
def set_student_info(info):
    global student_info
    student_info = info

conversation_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)