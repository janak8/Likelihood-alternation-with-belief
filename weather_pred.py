class BayesianPredictor:
    def __init__(self, prior_probs, conditional_probs):
        self.prior_probs = prior_probs
        self.conditional_probs = conditional_probs

    def predict(self, evidence):
        # Calculate the posterior probabilities using Bayes' theorem
        posterior_probs = {}
        evidence_sum = sum(self.conditional_probs[evidence].values())
        for outcome in self.prior_probs:
            likelihood = self.conditional_probs[evidence][outcome] / evidence_sum
            posterior_probs[outcome] = self.prior_probs[outcome] * likelihood

        # Normalize the posterior probabilities
        total_prob = sum(posterior_probs.values())
        for outcome in posterior_probs:
            posterior_probs[outcome] /= total_prob

        return posterior_probs


# Example usage:
if __name__ == "__main__":
    # Prior probabilities
    prior_probs = {"Rain": 0.15, "No Rain": 0.85}

    # Conditional probabilities
    conditional_probs = {
        "Cloudy": {"Rain": 0.80, "No Rain": 0.25},
        "Not Cloudy": {"Rain": 0.20, "No Rain": 0.75}
    }

    predictor = BayesianPredictor(prior_probs, conditional_probs)
    evidence = "Cloudy"
    posterior_probs = predictor.predict(evidence)

    print("Posterior probabilities after observing cloudy weather:", posterior_probs)

    if posterior_probs["Rain"] > posterior_probs["No Rain"]:
        print("Based on the cloudy weather, you should consider postponing the picnic.")
    else:
        print("Based on the cloudy weather, you can proceed with the picnic.")
