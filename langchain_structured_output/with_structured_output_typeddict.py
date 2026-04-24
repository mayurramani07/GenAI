from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatOpenAI()

# schema
class Review(TypedDict):

    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]
    

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I’ve been using the iPhone 15 for a couple of weeks now, and overall, it’s a solid upgrade. The A16 Bionic chip makes the phone extremely smooth—apps open instantly and multitasking feels effortless. The display is bright and color-accurate, which makes watching videos and browsing a great experience.

The camera performance is impressive, especially in daylight. Photos come out sharp with natural colors, and the portrait mode has improved significantly. Battery life easily lasts a full day with moderate usage, which is a big plus.

However, there are a few downsides. The charging speed still feels slow compared to other flagship phones, and the lack of major design changes makes it feel a bit repetitive. Also, the price is quite high for the features it offers.

Pros:
Excellent performance with A16 chip
Great camera quality
Reliable battery life

Cons:
Slow charging speed
Very expensive
Minimal design changes

Review by Mayur Ramani
""")

print(result['name'])