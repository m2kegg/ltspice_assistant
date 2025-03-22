import streamlit as st
import plotly.graph_objects as go
from plotly.graph_objects import Figure
import json
from langflow.load import run_flow_from_json
import os
import traceback
import sys

from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from rag_ltspice import build_knowledge_base, combine_analysis_with_rag, create_sample_circuit_knowledge, rag_enhanced_analysis
from tools_ltspice import analyze_frequency, analyze_transient, analyze_monte_carlo, generate_report
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langgraph.prebuilt import create_react_agent

 
st.set_page_config(
    page_title="Анализатор электронных схем",
    layout="wide",
    page_icon="🔌",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .main-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #4A90E2;
        }
        .stPlotlyChart {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.05);
            padding: 10px;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key="AIzaSyD7zUjKojFcpokb0v_JymHlNP6Pcs97V3U")

knowledge_base_path = "circuit_knowledge_base"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = FAISS.load_local(knowledge_base_path, embeddings, allow_dangerous_deserialization=True)
retrieval_qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Промпт для агента с RAG
prompt_template = PromptTemplate(
    template="""
Ты - AI-ассистент, специализирующийся на электронном проектировании и анализе схем.

Задача пользователя: {input}

Дополнительный контекст, связанный с этой темой: {context}

У тебя есть доступ к специализированной базе знаний с технической документацией по электронным схемам через систему RAG (Retrieval-Augmented Generation).

Ты можешь выполнять следующие типы анализа:
1. Анализ стабильности схем (запас по фазе, запас по усилению)
2. Анализ шумовых характеристик при разных условиях
3. Анализ переходных процессов
4. Частотный анализ (АЧХ и ФЧХ)
5. Анализ Монте-Карло для оценки влияния разброса параметров компонентов
6. Комплексный анализ с генерацией отчета
7. Поиск дополнительной информации в базе знаний

Все функции выполняются на стороне. Твоя задача - просто правильно отдать аргументы. И всё.
Используй соответствующие функции:
- analyze_noise() для шумового анализа
- analyze_transient() для анализа переходных процессов
- analyze_frequency(circuit_name ->  str (название файла LTSpice со схемой (обязательно включать расширение)), frequency_range -> (start_frequency, stop_frequency, counts) (частоты выбираешь сам и записываешь в кортеж в строго определённом формате. Например, (1, 1000000, 100)), parameters (дополнительные параметры, например, изменения нагрузки) -> словарь вида параметр - значение) для частотного анализа
- analyze_monte_carlo() для статистического анализа
- generate_report() для комплексного отчета
- rag_enhanced_analysis() для поиска информации в базе знаний
- combine_analysis_with_rag() для объединения результатов анализа с данными из базы знаний

Всегда объясняй результаты анализа понятным языком и дополняй их релевантной информацией из базы знаний.
""",
    input_variables=["input", "context"]
)

tools = [
    Tool(
        name="AnalyzeFrequency",
        func=analyze_frequency,
        description="Частотный анализ (АЧХ и ФЧХ)."
    ),
    Tool(
        name="AnalyzeTransient",
        func=analyze_transient,
        description="Анализ переходных процессов."
    ),
    Tool(
        name="AnalyzeMonteCarlo",
        func=analyze_monte_carlo,
        description="Анализ Монте-Карло для оценки влияния разброса параметров компонентов."
    ),
    Tool(
        name="GenerateReport",
        func=generate_report,
        description="Генерация комплексного отчета по результатам анализа."
    ),
]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True,
)

agent = create_react_agent(
    llm,
    [analyze_frequency, analyze_transient, analyze_monte_carlo, generate_report]
)


with st.sidebar:
    st.title("🔌 Анализатор электронных схем")
    
    agent_type = st.radio(
        "Выберите тип агента:",
        ["Автоматизация рутинных задач", "Комплексные аналитические отчеты"]
    )
    
    st.subheader("Настройки отображения")
    show_graphs = st.checkbox("Включить графики", value=True)
    show_tables = st.checkbox("Включить таблицы данных", value=True)
    
    plotly_theme = st.selectbox(
        "Тема графиков",
        ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"]
    )


st.title("Анализ электронных схем LTSpice")
tab1, tab2, tab3 = st.tabs(["📥 Ввод запроса", "📊 Результаты анализа", "📈 Визуализация данных"])

with tab1:
    st.header("📥 Ввод запроса")
    user_query = st.text_area(
        "Введите запрос для анализа схемы:",
        height=150,
        placeholder="Например: Проанализируй стабильность схемы при разных нагрузках."
    )
    uploaded_file = st.file_uploader(
        "Загрузите файл схемы (.asc):",
        type=["asc"],
        help="Файл должен быть в формате LTSpice."
    )
    analyze_button = st.button("Запустить анализ")
    

with tab2:
    st.header("📊 Результаты анализа")
    
    if 'results' not in st.session_state:
        st.info("Здесь будут отображены результаты анализа после выполнения.")
    else:
        st.markdown(st.session_state.results)
        
        st.download_button(
            label="Скачать отчет в PDF",
            data=st.session_state.results.encode('utf-8'),
            file_name="circuit_analysis_report.pdf",
            mime="application/pdf"
        )

with tab3:
    st.header("📈 Визуализация данных")
    if 'visualization_data' not in st.session_state:
        st.info("Графики будут отображены после выполнения анализа.")
    else:
        if 'plot_json' in st.session_state.visualization_data:
            plot_data = json.loads(st.session_state.visualization_data['plot_json'])
            ach_fig = Figure(json.loads(plot_data['ach']))
            fch_fig = Figure(json.loads(plot_data['fch']))
            ach_fig.update_layout(template=plotly_theme)
            fch_fig.update_layout(template=plotly_theme)
            st.subheader("Амплитудно-частотная характеристика")
            st.plotly_chart(ach_fig, use_container_width=True)
            st.subheader("Фазо-частотная характеристика")
            st.plotly_chart(fch_fig, use_container_width=True)
        
if analyze_button and user_query and uploaded_file:
    with st.spinner("Выполняется анализ..."):
        try:
            file_path = f"./circuits/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            circuit_name = uploaded_file.name.split('.')[0]
            user_query = user_query + f"Название схемы: {uploaded_file.name}"
            context = retrieval_qa_chain.run(user_query)
            input_message = prompt_template.format(input=user_query, context=context)
            analysis_results = agent_executor.run(input_message)
            rag_results = rag_enhanced_analysis(user_query)
            combined_report = combine_analysis_with_rag(analysis_results, user_query, circuit_name)
            st.success("Анализ завершен!")
            st.text_area("Результаты анализа:", value=combined_report, height=300)

        except Exception as e:
            # Получаем полную информацию об ошибке
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_details = traceback.format_exception(exc_type, exc_value, exc_traceback)
            
            # Отображаем краткую информацию об ошибке
            st.error(f"Ошибка при выполнении анализа: {str(e)}")
            
            # Отображаем подробную информацию в развернутом блоке
            with st.expander("Подробная информация об ошибке"):
                st.code(''.join(error_details), language="python")
                
            # Логируем ошибку для отладки
            print(''.join(error_details))