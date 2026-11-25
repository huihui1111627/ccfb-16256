"""Flask 应用: 页面 + JSON API。"""
from __future__ import annotations

import os
from flask import Flask, jsonify, render_template, request

from .service import SERVICE


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["JSON_AS_ASCII"] = False

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/overview")
    def api_overview():
        return jsonify(SERVICE.overview())

    @app.route("/api/heatmap")
    def api_heatmap():
        return jsonify(SERVICE.heatmap(max_rows=int(request.args.get("rows", 900)),
                                       bins=int(request.args.get("bins", 420))))

    @app.route("/api/distribution")
    def api_distribution():
        return jsonify(SERVICE.distribution())

    @app.route("/api/hierarchy")
    def api_hierarchy():
        return jsonify(SERVICE.hierarchy())

    @app.route("/api/incidents")
    def api_incidents():
        return jsonify(SERVICE.incidents())

    @app.route("/api/nodes")
    def api_nodes():
        return jsonify(SERVICE.node_list(request.args.get("cause"),
                                         int(request.args.get("limit", 200)),
                                         include_ok=request.args.get("all") == "1"))

    @app.route("/api/node/<int:idx>")
    def api_node(idx):
        return jsonify(SERVICE.node_detail(idx))

    @app.route("/api/leap")
    def api_leap():
        return jsonify(SERVICE.leap_data())

    @app.route("/api/scan")
    def api_scan():
        return jsonify(SERVICE.scan_data())

    @app.route("/api/advice")
    def api_advice():
        return jsonify(SERVICE.advice())

    @app.route("/api/eval")
    def api_eval():
        return jsonify(SERVICE.eval)

    return app


app = create_app()


def main():
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5199)), debug=False)


if __name__ == "__main__":
    main()
