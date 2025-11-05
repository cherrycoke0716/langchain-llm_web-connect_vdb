from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. API 키 로드 (.env 파일에서)
load_dotenv()

# 2. LLM 모델 정의
# (이 부분만 실행해도 API 키가 잘 작동하는지 테스트됨)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 3. (중요) RAG용 프롬프트 템플릿 "뼈대" 정의
# 나중에 벡터 DB에서 가져온 {context}를 여기에 삽입할 것입니다.
RAG_PROMPT_TEMPLATE = """
당신은 질문에 답변하는 AI 어시스턴트입니다.
제공된 "문맥"을 바탕으로 사용자의 "질문"에 정확하게 답변해 주세요.

[문맥]
{context}

[질문]
{question}

[답변]
"""

def get_basic_chain():
    """
    (테스트용) LLM이 잘 작동하는지 확인하는 기본 체인
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ])
    
    # LangChain Expression Language (LCEL)로 체인 묶기
    chain = prompt | llm | StrOutputParser()
    return chain

def get_rag_chain():
    """
    (뼈대만 구현) 나중에 완성할 RAG 체인
    """
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # TODO: 벡터 DB가 준비되면 여기에 'retriever'가 추가되어야 함
    # chain = (
    #    {"context": retriever, "question": RunnablePassthrough()}
    #    | prompt
    #    | llm
    #    | StrOutputParser()
    # )
    
    print("RAG 체인 뼈대 준비 완료. (retriever 연결 필요)")
    
    # 지금은 임시로 프롬프트만 반환
    return prompt


# --- 이 파일을 직접 실행해서 테스트 ---
if __name__ == "__main__":
    print("LLM 기본 체인 테스트...")
    basic_chain = get_basic_chain()
    answer = basic_chain.invoke({"question": "LangChain이 뭐야?"})
    
    print("\n--- LLM 답변 ---")
    print(answer)