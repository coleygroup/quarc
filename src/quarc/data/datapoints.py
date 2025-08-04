from dataclasses import dataclass
import pandas as pd


@dataclass
class AgentRecord:
    smiles: str
    amount: float | None


@dataclass
class ReactionDatum:
    """Represents a reaction datum.

    Attributes:
        document_id (str): The document ID of the reaction.
        rxn_class (str): The name of the reaction.
        date (str): The date of the patent.
        rxn_smiles (str): The atom-mapped reaction SMILES.
        reactants (list[AgentRecord]): A list of AgentRecord objects with smiles and amount information for reactants.
        products (list[AgentRecord]): A list of AgentRecord objects with smiles and amount information for products.
        agents (list[AgentRecord]): A list of AgentRecord objects with smiles and amount information for agents.
        temperature (float | None): The target temperature, which could be None.
    """

    # Reaction Info
    document_id: str | None
    rxn_class: str | None
    date: str | None
    rxn_smiles: str

    # Reaction Context
    reactants: list[AgentRecord]
    products: list[AgentRecord]
    agents: list[AgentRecord]

    # Targets
    temperature: float | None


def convert_reaction_data_to_dataframe(data: list[ReactionDatum]) -> pd.DataFrame:
    rows = []
    for reaction in data:
        row = {
            "document_id": reaction.document_id,
            "rxn_class": reaction.rxn_class,
            "date": reaction.date,
            "rxn_smiles": reaction.rxn_smiles,
            "temperature": reaction.temperature,
        }

        # Process reactants
        for i in range(5):
            if i < len(reaction.reactants):
                row[f"Reactant_{i+1}_smi"] = reaction.reactants[i].smiles
                row[f"Reactant_{i+1}_mol"] = reaction.reactants[i].amount
            else:
                row[f"Reactant_{i+1}_smi"] = None
                row[f"Reactant_{i+1}_mol"] = None

        # Process agents (similar to reactants)
        for i in range(5):
            if i < len(reaction.agents):
                row[f"Agent_{i+1}_smi"] = reaction.agents[i].smiles
                row[f"Agent_{i+1}_mol"] = reaction.agents[i].amount
            else:
                row[f"Agent_{i+1}_smi"] = None
                row[f"Agent_{i+1}_mol"] = None

        # Process products (similar to reactants)
        row["product_smi"] = reaction.products[0].smiles
        row["product_mol"] = reaction.products[0].amount

        rows.append(row)

    return pd.DataFrame(rows)
