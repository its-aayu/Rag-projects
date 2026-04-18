def generate_answer(client, query, context_docs):
    context = "\n\n".join(
        f"[Context {index}]\n{doc.page_content}"
        for index, doc in enumerate(context_docs, start=1)
    )

    prompt = f"""
You are an AI assistant.

Answer the question ONLY using the provided context.
The context is ranked from most relevant to least relevant.
Prefer the first context that directly answers the question.
Ignore unrelated context chunks.
If the answer is not in context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    return response.choices[0].message.content
