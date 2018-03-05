from flask import Flask, request, render_template
from lightoff import single_solve

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('q.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['textbox']
    try:
        processed_text = single_solve(text)
    except:
        processed_text = "Error"
    #processed_text = text.replace('\n', '<br>')
    return render_template('q.html', answer=processed_text, question=text)

app.run()
