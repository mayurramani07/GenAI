from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()

model2 = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Large Language Models (LLMs) are advanced AI systems trained on massive datasets to understand and generate human-like text.

They are based on transformer architecture, which uses attention mechanisms to process input data efficiently.

Key advantages of LLMs include:
- Ability to perform multiple tasks like summarization, translation, and question answering.
- Context understanding across long text sequences.
- Can be fine-tuned or used with prompting techniques.

However, LLMs also have limitations:
- They may generate incorrect or hallucinated information.
- Require large computational resources.
- Sensitive to prompt design.

Popular examples of LLMs include GPT models, Claude, and open-source models like LLaMA.

LLMs are widely used in chatbots, content generation, coding assistants, and AI-powered search systems.
"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()
