from flask import Flask, request, jsonify, json, render_template

from lightsoff import turnoff_lights

app = Flask(__name__)

@app.route("/healthCheck")
def index():
    return "ok"


@app.route("/", methods=['POST'])
def lightsoff():
    #print(request)
    req_data = request.get_json()
    #print(req_data)
    try:
        solution = turnoff_lights(req_data)
        if solution is not None:
            # convert from numpy array to normal
            solution = [[int(x) for x in row] for row in solution]
            print(solution)
        return json.dumps({'status': 'success', 'answer': solution})
    except Exception as e:
        return json.dumps({'status': 'failed', 'reason': str(e)})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
