import ollama
def ask(prompt):
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]



# Zero shot 

prompt = """
A pencil costs $3.
How much do 7 pencils cost?
"""

print(ask(prompt))


# One Shot 

prompt = """
Solve the math problem.

Example:
A notebook costs $5.
If you buy 4 notebooks:
5 × 4 = $20

Now solve this:
A pencil costs $3.
If you buy 7 pencils:
"""

print(ask(prompt))

# Few Shot

prompt = """
Solve the following problems.

Problem:
An apple costs $2.
If you buy 5 apples:
2 × 5 = $10

Problem:
A pen costs $4.
If you buy 3 pens:
4 × 3 = $12

Problem:
A pencil costs $3.
If you buy 7 pencils:
"""

print(ask(prompt))




# Zero Shot Chain Of Thought

prompt = """
A pencil costs $3.
How much do 7 pencils cost?

Explain your reasoning step by step.
"""

print(ask(prompt))