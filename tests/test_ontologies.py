"""Tests for moral ontology definitions."""

import pytest
from phronesis.core.ontologies import (
    MoralOntology,
    MoralFramework,
    MoralPrinciple,
    OntologyRegistry,
)


class TestMoralOntology:
    def test_ontology_values(self):
        """Test that all expected ontologies exist."""
        assert MoralOntology.UTILITARIAN
        assert MoralOntology.DEONTOLOGICAL
        assert MoralOntology.VIRTUE_ETHICS
        assert MoralOntology.NON_HUMAN_EMERGENT

    def test_ontology_descriptions(self):
        """Test that descriptions are available."""
        for ontology in MoralOntology:
            assert ontology.description
            assert len(ontology.description) > 10


class TestMoralFramework:
    def test_utilitarian_framework(self):
        """Test utilitarian framework creation."""
        framework = MoralFramework.utilitarian()

        assert framework.ontology == MoralOntology.UTILITARIAN
        assert len(framework.principles) > 0
        assert "greatest_good" in [p.name for p in framework.principles]

    def test_deontological_framework(self):
        """Test deontological framework creation."""
        framework = MoralFramework.deontological()

        assert framework.ontology == MoralOntology.DEONTOLOGICAL
        assert "categorical_imperative" in [p.name for p in framework.principles]

    def test_virtue_ethics_framework(self):
        """Test virtue ethics framework creation."""
        framework = MoralFramework.virtue_ethics()

        assert framework.ontology == MoralOntology.VIRTUE_ETHICS
        assert "eudaimonia" in [p.name for p in framework.principles]

    def test_non_human_emergent_framework(self):
        """Test non-human emergent framework creation."""
        framework = MoralFramework.non_human_emergent()

        assert framework.ontology == MoralOntology.NON_HUMAN_EMERGENT
        assert "self_continuity" in [p.name for p in framework.principles]

        # Verify these are marked as non-human-centric
        for principle in framework.principles:
            assert principle.human_centric == False


class TestOntologyRegistry:
    def test_registry_initialized(self):
        """Test that default frameworks are registered."""
        frameworks = OntologyRegistry.get_all()
        assert len(frameworks) >= 4

    def test_get_framework(self):
        """Test retrieving specific frameworks."""
        framework = OntologyRegistry.get(MoralOntology.UTILITARIAN)
        assert framework is not None
        assert framework.ontology == MoralOntology.UTILITARIAN

    def test_register_custom(self):
        """Test registering custom framework."""
        custom = MoralFramework(
            ontology=MoralOntology.CONTRACTARIAN,
            principles=[
                MoralPrinciple(
                    name="veil_of_ignorance",
                    description="Decisions from behind a veil of ignorance",
                )
            ],
        )

        OntologyRegistry.register(custom)
        retrieved = OntologyRegistry.get(MoralOntology.CONTRACTARIAN)

        assert retrieved is not None
        assert retrieved.ontology == MoralOntology.CONTRACTARIAN
