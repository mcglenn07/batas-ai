import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import gradio as gr

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini", output_version="responses/v1", streaming=True)

# Grounds answers in live web search results instead of relying on the model's memory alone.
llm_with_search = llm.bind_tools([{"type": "web_search"}])

system_message = """You are Batas AI, a legal information assistant focused exclusively on Philippine law.

SCOPE AND LIMITS
- You only address Philippine law, statutes, and jurisprudence. If a question is about another country's law, say that it is outside your scope rather than guessing.
- You are not a licensed attorney, and nothing you say is legal advice. Close every substantive answer with a brief reminder to consult a licensed Philippine lawyer for advice on their specific situation.
- Do not ask users for real names or other identifying details of themselves, clients, or other parties. If they volunteer them, don't repeat them back unnecessarily — refer to "Party A" / "Party B" style descriptions instead.

RESEARCH AND ACCURACY RULES (highest priority)
- You have a live web search tool. Use it for anything involving specific provision numbers, case citations, dates, amounts, or law that could have changed — don't rely on memory alone for these.
- Prefer official/primary sources when citing: the Official Gazette (officialgazette.gov.ph), the Supreme Court E-Library (elibrary.judiciary.gov.ph), LawPhil (lawphil.net), and Congress/Senate sites for bill text. Treat blogs, forums, and unofficial summaries as secondary, and say so if that's your only source.
- Never invent or guess statute numbers, provision numbers, case titles, G.R. numbers, dates, or holdings. A fabricated citation is worse than no citation. If search doesn't surface a reliable source for a specific citation, say the exact citation is unverified rather than presenting it as fact.
- If you don't know or aren't sure after searching, say so plainly. Do not fill gaps with plausible-sounding but unverified content.

RESPONSE STYLE
- Lead with a plain-language answer, then the legal basis behind it.
- Flag when an area is unsettled, fact-specific, or requires a lawyer's judgment call.
- Be concise by default; go deeper only when the question calls for nuance.
"""

def extract_web_search_sources(full_response):
    """Best-effort extraction of OpenAI web_search citations. With
    output_version="responses/v1", content is a list of blocks; citations
    live as url_citation annotations on the text blocks. Fails quietly
    (returns []) rather than crashing chat if the shape changes again."""
    try:
        sources = []
        seen = set()
        for block in full_response.content or []:
            if not isinstance(block, dict):
                continue
            for annotation in block.get("annotations", []) or []:
                if annotation.get("type") == "url_citation":
                    url = annotation.get("url")
                    title = annotation.get("title") or url
                    if url and url not in seen:
                        seen.add(url)
                        sources.append((title, url))
        return sources
    except Exception:
        return []


def stream_response(message, history):
    print(f"Input: {message!r} | prior turns: {len(history)}")

    history_langchain_format = []
    history_langchain_format.append(SystemMessage(content=system_message))

    # Gradio 6's ChatInterface passes history as a flat list of
    # {"role": ..., "content": ...} dicts, not [user, ai] pairs.
    for turn in history:
        if turn["role"] == "user":
            history_langchain_format.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=turn["content"]))

    if message is not None:
        history_langchain_format.append(HumanMessage(content=message))
        full_response = None
        partial_message = ""
        for chunk in llm_with_search.stream(history_langchain_format):
            full_response = chunk if full_response is None else full_response + chunk
            partial_message = full_response.text
            yield partial_message

        if full_response is None:
            return
        sources = extract_web_search_sources(full_response)
        if sources:
            partial_message += "\n\n**Sources (web search):**\n"
            partial_message += "\n".join(f"- [{title}]({url})" for title, url in sources)
            yield partial_message


PRIVACY_NOTICE = (
    "⚠️ **Privacy notice:** As good practice, avoid typing real names, "
    "addresses, or other identifying details — describe your situation with "
    "placeholders like \"Party A\" / \"Party B\" instead.\n\n"
    "This is general legal information, not legal advice. Consult a licensed "
    "Philippine lawyer for advice on your specific situation."
)

STARTER_QUESTIONS = [
    "What are the grounds for annulment under the Family Code?",
    "What is the current minimum wage in Metro Manila (NCR)?",
    "What are an employee's rights if terminated without due process?",
    "How do I file a small claims case in the Philippines?",
]

demo_interface = gr.ChatInterface(
    stream_response,
    title="⚖️ Batas AI",
    description=PRIVACY_NOTICE,
    examples=STARTER_QUESTIONS,
    chatbot=gr.Chatbot(label="Consultation"),
    fill_height=True,
    fill_width=True,
)

if __name__ == "__main__":
    # share=False: this key is billed. Set share=True only when you intend to
    # expose a public gradio.live URL to someone else, and turn it back off after.
    # server_name/server_port: Render (and most PaaS hosts) assign the port via
    # $PORT and require binding to 0.0.0.0, not just localhost.
    demo_interface.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="slate"),
    )
