import os
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
from typing import List
from brain import get_embedding, format_docs, calculate_cosine_similarity_parallel, calculate_cosine_similarity_batch
from connectdb import connect_to_postgresql, query_qas, load_chroma, query_qas_details
from data_preprocessing import tien_xu_li
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import time
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, chain
from pydantic import BaseModel, Field
import logging
import timeit

# Environment setup
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
load_dotenv()
logging.basicConfig(level=logging.DEBUG)

class Search(BaseModel):
    """Search over a database of job records."""
    queries: List[str] = Field(
        ...,
        description="Truy vấn riêng biệt để tìm kiếm",
    )

def retriever_df(llm):
    SYSTEM_TEMPLATE = """
       Trả lời câu hỏi người dùng dựa trên ngữ cảnh bên dưới và thêm nguồn tham khảo bên dưới.
       Nếu ngữ cảnh không chứa bất cứ thông tin liên quan đến câu hỏi, đừng làm gì hết chỉ cần trả lời 
       "Xin lỗi, hiện tại tôi không thể trả lời câu hỏi của bạn.":

       <context>
       {context}
       </context>
       <citations>
       {source}
       </citations>
       """
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)
    return document_chain

def initialize():
    # Database connection
    connection = connect_to_postgresql()
    df = None
    if connection:
        df = query_qas()
        connection.close()
    else:
        logging.error("Failed to connect to PostgreSQL.")
        return None

    # OpenAI API key setup
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-0125",
        temperature=0,
        max_tokens=4000
    )
    output_parser = PydanticToolsParser(tools=[Search])

    system = """Bạn có khả năng đưa ra các truy vấn tìm kiếm từ việc tách nhỏ câu hỏi nếu có nhiều ý thành những câu hỏi riêng lẻ
             giúp thông tin rõ ràng và quá trình tìm kiếm tốt hơn. 
            Nếu cần tra cứu hai thông tin riêng biệt, bạn được phép làm điều đó!
            Nếu không thể tách ra thì hãy trả về câu cũ"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    structured_llm = llm.with_structured_output(Search)
    query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

    vectorstore = load_chroma()
    if vectorstore is None:
        logging.error("Failed to load Chroma vectorstore.")
        return None

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    contextualize_q_system_prompt = """Đưa ra lịch sử trò chuyện và câu hỏi mới nhất của người dùng \
            có thể tham chiếu ngữ cảnh trong lịch sử trò chuyện, tạo thành một câu hỏi độc lập \
            có thể hiểu được nếu không có lịch sử trò chuyện. KHÔNG trả lời câu hỏi, \
            chỉ cần định dạng lại nó nếu cần và nếu không thì trả lại như cũ."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    SYSTEM_TEMPLATE = """
                   Trả lời câu hỏi người dùng dựa trên ngữ cảnh bên dưới và thêm nguồn tham khảo bên dưới nếu có trong lĩnh vực an toàn giao thông tại Việt Nam .
                    Nếu ngữ cảnh không chứa bất cứ thông tin liên quan đến câu hỏi, đừng làm gì hết chỉ cần trả lời 
                    "Xin lỗi, hiện tại tôi không thể trả lời câu hỏi của bạn.":

                   Câu hỏi: {question}
                   Context: {context}
                   <citations>
                   {source}
                   </citations>
                   Trả lời:
                   """
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
            RunnablePassthrough.assign(
                context=contextualize_q_chain | retriever | format_docs
            )
            | question_answering_prompt
            | llm
            | StrOutputParser()
    )

    return {
        "df": df,
        "llm": llm,
        "query_analyzer": query_analyzer,
        "retriever": retriever,
        "rag_chain": rag_chain,
        "contextualize_q_chain": contextualize_q_chain
    }

def process_question(question, chat_history, components):
    df = components["df"]
    llm = components["llm"]
    query_analyzer = components["query_analyzer"]
    retriever = components["retriever"]
    rag_chain = components["rag_chain"]
    contextualize_q_chain = components["contextualize_q_chain"]

    handle_question = tien_xu_li(question)
    chain_response = custom_chain(handle_question, query_analyzer, retriever)
    if len(chain_response[0].queries) > 1:
        handle_question = " ".join(chain_response[0].queries)
        logging.debug(chain_response[0].queries)
    else:
        handle_question = handle_question
    logging.debug("handle_question: %s", handle_question)
    
    # docs = []
    # for query in chain_response[0].queries:
    #     new_docs = retriever.invoke(query)
    #     docs.extend(new_docs)

    start_time = timeit.default_timer()
    # Embedding and cosine similarity
    embed_query = np.array(get_embedding().embed_query(handle_question))
    batch_size = 1000
    cosine_similarities = []
    for i in range(0, len(df), batch_size):
        batch_titles = df['embedding_title'][i:i + batch_size]
        batch_similarities = calculate_cosine_similarity_batch(batch_titles, embed_query)
        cosine_similarities.extend(batch_similarities)

    df['cosine_similarity'] = cosine_similarities

    elapsed_time = timeit.default_timer() - start_time
    logging.debug("Embedding and df time: %s", elapsed_time)

    max_cosine_row = df.loc[df['cosine_similarity'].idxmax()]
    result = ""
    response = []
    source = []
    logging.debug(max_cosine_row)

    if max_cosine_row['cosine_similarity'] > 0.75:
        detailed_df = query_qas_details([max_cosine_row['id']])
        detailed_row = detailed_df.iloc[0]
        doc = Document(page_content=detailed_row['content'], metadata={"source": detailed_row['source']})

        source = [doc.metadata['source']]
        # logging.debug(doc)
        start_time = timeit.default_timer()
        ai_msg = retriever_df(llm).stream(
            {"question": handle_question,
             "source": source,
             "chat_history": chat_history,
             "context": [doc]}  # Ensure context is a list
        )
        elapsed_time = timeit.default_timer() - start_time
        logging.debug("retrieval time: %s", elapsed_time)
        for text in ai_msg:
            response.append(text)
            result = "".join(response).strip()
    else:
        doc = chain_response[1]
        # logging.debug(doc)
        start_time = timeit.default_timer()
        ai_msg = rag_chain.stream(
            {"question": handle_question,
            "source": source,
            "chat_history": chat_history,
            "context": doc}  # Ensure context is a list
        )
        
        elapsed_time = timeit.default_timer() - start_time
        logging.debug("retrieval time: %s", elapsed_time)
        for text in ai_msg:
            response.append(text)
            result = "".join(response).strip()

    chat_history.extend([HumanMessage(content=question), AIMessage(content=result)])
    # logging.debug("Assistant: %s", result)

    return result, elapsed_time, doc

def custom_chain(question, query_analyzer, retriever):
    response = query_analyzer.invoke(question)  # Phân tích câu hỏi và tạo các truy vấn con.
    docs = []  # Khởi tạo danh sách tài liệu rỗng.

    # Lặp qua từng truy vấn được tạo từ câu hỏi.
    for query in response.queries:
        new_docs = retriever.invoke(query)  # Truy xuất tài liệu liên quan.
        docs.extend(new_docs)  # Thêm các tài liệu truy xuất được vào danh sách.

    return [response, docs]  # Trả về kết quả phân tích và tài liệu.