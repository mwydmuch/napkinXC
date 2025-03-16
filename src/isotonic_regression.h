#include <vector>
#include <algorithm>
#include <numeric>

/*
class IsotonicRegression {
public:
    IsotonicRegression() = default;
    IsotonicRegression(

    // Fit the isotonic regression model
    void fit(const std::vector<float>& predictions, const std::vector<float>& targets) {
        if (predictions.empty() || predictions.size() != targets.size()) {
            throw std::invalid_argument("Invalid input sizes");
        }

        // Create initial blocks and sort by prediction value
        std::vector<size_t> indices(predictions.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                 [&predictions](size_t i1, size_t i2) {
                     return predictions[i1] < predictions[i2];
                 });
        
        fitted_blocks_.clear();
        fitted_blocks_.reserve(predictions.size());
        for (size_t idx : indices) {
            fitted_blocks_.emplace_back(predictions[idx], targets[idx]);
        }

        // Pool Adjacent Violators Algorithm
        bool violation_exists = true;
        while (violation_exists) {
            violation_exists = false;
            
            for (size_t i = 0; i < fitted_blocks_.size() - 1; ++i) {
                if (fitted_blocks_[i].target > fitted_blocks_[i + 1].target) {
                    // Merge blocks that violate monotonicity
                    Point merged = mergeBlocks(fitted_blocks_[i], fitted_blocks_[i + 1]);
                    fitted_blocks_[i] = merged;
                    fitted_blocks_[i + 1] = merged;
                    violation_exists = true;
                }
            }
            
            // Remove duplicate blocks after merging
            auto new_end = std::unique(fitted_blocks_.begin(), fitted_blocks_.end());
            fitted_blocks_.erase(new_end, fitted_blocks_.end());
        }

        is_fitted_ = true;
    }

    // Transform new data using the fitted model
    std::vector<float> transform(const std::vector<float>& predictions) const {
        if (!is_fitted_) {
            throw std::runtime_error("Model must be fitted before transform");
        }

        std::vector<float> transformed(predictions.size());
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            const float pred = predictions[i];
            
            // Find the appropriate blocks for interpolation
            auto it = std::lower_bound(
                fitted_blocks_.begin(), fitted_blocks_.end(), pred,
                [](const Point& p, float val) { return p.pred < val; }
            );
            
            if (it == fitted_blocks_.end()) {
                // If beyond the last block, use the last target value
                transformed[i] = fitted_blocks_.back().target;
            } else if (it == fitted_blocks_.begin()) {
                // If before the first block, use the first target value
                transformed[i] = fitted_blocks_.front().target;
            } else {
                // Interpolate between blocks
                transformed[i] = interpolate(*(it - 1), *it, pred);
            }
        }
        
        return transformed;
    }

    // Fit and transform in one step
    std::vector<float> fit_transform(const std::vector<float>& predictions,
                                   const std::vector<float>& targets) {
        fit(predictions, targets);
        return transform(predictions);
    }

    // Check if the model has been fitted
    bool is_fitted() const {
        return is_fitted_;
    }

    // Get the fitted blocks (for debugging or analysis)
    std::vector<std::pair<float, float>> get_fitted_points() const {
        if (!is_fitted_) {
            throw std::runtime_error("Model must be fitted first");
        }
        std::vector<std::pair<float, float>> points;
        points.reserve(fitted_blocks_.size());
        for (const auto& block : fitted_blocks_) {
            points.emplace_back(block.pred, block.target);
        }
        return points;
    }

private:
    // Helper struct to store prediction, target, and weight
    struct Point {
        float pred;
        float target;
        float weight;
        
        Point(float p, float t, float w = 1.0f) 
            : pred(p), target(t), weight(w) {}

        bool operator==(const Point& other) const {
            return target == other.target;
        }
    };
    
    std::vector<Point> fitted_blocks_;
    bool is_fitted_ = false;

    // Helper function to merge two blocks
    static Point mergeBlocks(const Point& p1, const Point& p2) {
        float total_weight = p1.weight + p2.weight;
        float weighted_avg = (p1.target * p1.weight + p2.target * p2.weight) / total_weight;
        return Point(p1.pred, weighted_avg, total_weight);
    }

    // Helper function to interpolate between two points
    static float interpolate(const Point& left, const Point& right, float x) {
        if (left.pred == right.pred) return left.target;
        float t = (x - left.pred) / (right.pred - left.pred);
        return left.target * (1 - t) + right.target * t;
    }
};
*/