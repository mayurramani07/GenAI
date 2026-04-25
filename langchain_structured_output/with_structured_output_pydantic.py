from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

# schema
class Review(BaseModel):

    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")
    

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I’ve been using the Apple MacBook Air M3 for the past two weeks, and overall it’s been a smooth experience. The M3 chip is incredibly fast for daily tasks like coding, browsing, and even light video editing. Apps open instantly, and there’s almost no lag even with multiple tabs and software running together.

The battery life is honestly one of the best I’ve seen—it easily lasts more than a full day on moderate usage. The build quality feels premium as always, and the lightweight design makes it perfect for carrying around.

However, the base model with 8GB RAM feels limiting, especially when running heavy applications. Also, the lack of ports is still frustrating—you basically need a dongle for everything. The price is also on the higher side compared to other laptops with similar specs.

Pros:
Excellent performance with M3 chip
Outstanding battery life
Lightweight and premium design

Cons:
Limited RAM in base variant
Very few ports
Expensive

Review by Mayur Ramani
""")

print(result) 