import json


class Preprocess:
    def preprocess(self, body, state: dict, collect_custom_statistics_fn=None):
        if isinstance(body, bytes):
            body = json.loads(body)
        return [body.get("text", "")]

    def postprocess(self, data, state: dict, collect_custom_statistics_fn=None):
        label = data[0] if hasattr(data, "__iter__") else data
        return {"label": str(label)}
