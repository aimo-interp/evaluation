import json
import argparse
import os
import webbrowser
import mistune
import re

def chat_style_renderer(file_path):
    if not os.path.exists(file_path):
        print(f"File not found.")
        return

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body);"></script>
        <style>
            body { font-family: 'Inter', -apple-system, sans-serif; background: #f9f9f9; padding: 50px; }
            .chat-bubble { 
                background: white; border: 1px solid #e5e5e5; padding: 25px; 
                margin-bottom: 20px; border-radius: 15px; max-width: 850px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05); line-height: 1.6;
            }
            .label { font-weight: bold; color: #555; margin-bottom: 10px; display: block; }
            .math-block { overflow-x: auto; padding: 10px 0; }
            h3 { color: #222; margin-top: 25px; }
        </style>
    </head>
    <body>
        <h2 style="text-align:center">GPT-5 Reasoning Trace</h2>
    """

    content = ""
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            raw_trace = data["model_response"]

            # STEP 1: Fix the JSON escaping
            # This is what Chatbots do: they normalize the backslashes
            trace = raw_trace.replace('\\\\', '\\')

            # STEP 2: Convert common LLM delimiters to KaTeX-friendly ones
            # KaTeX is very picky. It loves $ and $$
            trace = trace.replace('\\(', '$').replace('\\)', '$')
            trace = trace.replace('\\[', '$$').replace('\\]', '$$')

            # STEP 3: Convert Markdown to HTML (Handling the ### headers)
            markdown = mistune.create_markdown()
            html_trace = markdown(trace)

            content += f"""
            <div class="chat-bubble">
                <span class="label">RECORD #{i+1} | Result: {"✅" if data['is_correct'] else "❌"}</span>
                <div class="math-block">
                    {html_trace}
                </div>
            </div>
            """

    full_html = html_template + content + """
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: "$$", right: "$$", display: true},
                    {left: "$", right: "$", display: false},
                    {left: "\\\\(", right: "\\\\)", display: false},
                    {left: "\\\\[", right: "\\\\]", display: true}
                ]
            });
        });
    </script>
    </body></html>"""

    with open("chat_view.html", "w", encoding="utf-8") as f:
        f.write(full_html)
    
    webbrowser.open('file://' + os.path.realpath("chat_view.html"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    chat_style_renderer(args.path)