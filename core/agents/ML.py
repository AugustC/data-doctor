from typing import Literal

from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from core.agents import State
from core.ml import Classifier, Regression
from core.utils import get_columns, cleanup_data, normalize_data
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from core.ml.BaseMLModels import BaseMLModels
import pandas as pd

class Data(BaseModel):
    target: Literal["chronic_obstructive_pulmonary_disease", "alanine_aminotransferases"]
    data: dict

def process_data(state: State) -> State:
    message = state.get("messages", [])[-1]
    sys_prompt = (
        "You are a data processing agent. Extract from the user's request the necessary data for prediction.\n"
        "# Rules\n"
        "1. Extract all variables. If not mentioned, include it as a null value.\n"
        "# Necessary data includes:\n"
        "{columns}\n"
        "# User's request\n"
        "{message}\n"
        "# JSON Output Format\n"
        "{{\n"
        "   \"target\": \"<target variable. Must be one of ['chronic_obstructive_pulmonary_disease', 'alanine_aminotransferases'>\",\n"
        "   \"data\": {{\n"
        "       \"<column_name>\": <value>\n"
        "   }}\n"
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        timeout=10
    )
    prompt = PromptTemplate(
        template=sys_prompt,
        input_variables=["columns", "message"],
    )
    chain = prompt | llm.with_structured_output(
        Data, method="json_mode"
    )
    response: Data = chain.invoke({
        "columns": get_columns(),
        "message": message
    })
    return {
        "target": response.target,
        "data": response.data
    }

def predict_classifier(state: State) -> State:
    classifier = Classifier()
    classifier.load_model("models/classifier.model")
    instance = pd.DataFrame(state.get("data", {}), index=[0])
    instance = cleanup_data(instance)
    instance = normalize_data(instance)
    classifier_prediction = classifier.predict(instance)
    return {
        "prediction": classifier_prediction,
    }

def predict_regressor(state: State) -> State:
    regressor = Regression()
    regressor.load_model("models/regression.model")
    instance = pd.DataFrame(state.get("data", {}), index=[0])
    instance = cleanup_data(instance)
    instance = normalize_data(instance)
    regressor_prediction = regressor.predict(instance)
    return {
        "prediction": regressor_prediction,
    }

def prediction_type(state: State) -> Literal["classifier", "regressor", END]:
    if state.get("target", "") == "chronic_obstructive_pulmonary_disease":
        return "classifier"
    elif state.get("target", "") == "alanine_aminotransferases":
        return "regressor"
    else:
        return END


class ML:
    """
    Machine Learning agent that processes data and provides predictions.
    """

    def __init__(self):
        super().__init__()
        self.name = "MLAgent"
        self.description = (
                "This agent generates predictions for the input data using machine learning models."
                "This agent must be used for prediction of chronic obstructive pulmonary disease or alanine aminotransferases."
        )

    def build_graph(self):
        graph = StateGraph(State)
        graph.add_node("process_data", process_data)
        graph.add_node("classifier", predict_classifier)
        graph.add_node("regressor", predict_regressor)
        graph.add_edge(START, "process_data")
        graph.add_conditional_edges("process_data", prediction_type)
        graph.add_edge("classifier", END)
        graph.add_edge("regressor", END)
        return graph.compile()

