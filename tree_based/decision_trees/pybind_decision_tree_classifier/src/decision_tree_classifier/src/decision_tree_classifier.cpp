#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> // Include Eigen support for zero-copy with NumPy
#include <pybind11/stl.h>   // For STL containers like std::vector
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <memory>      // For smart pointers
#include <numeric>     // For std::iota
#include <string_view> // For string_view

namespace py = pybind11;

// Define criterion types as enum for type safety and better performance
enum class Criterion
{
    Gini,
    Entropy
};

// Constants for default hyperparameters
constexpr int DEFAULT_MAX_DEPTH = 3;
constexpr int DEFAULT_MIN_SAMPLES_SPLIT = 2;
constexpr double DEFAULT_MIN_IMPURITY_DECREASE = 1e-7;

//
// A node in the decision tree using smart pointers
//
struct TreeNode
{
    int feature_index;               // Index of the feature to split on
    double threshold;                // Threshold value for the split
    std::unique_ptr<TreeNode> left;  // Left child node
    std::unique_ptr<TreeNode> right; // Right child node
    int label;                       // Used if this node is a leaf
    bool is_leaf;                    // True if this node is a leaf node

    TreeNode() noexcept : feature_index(-1), threshold(0.0), label(-1), is_leaf(false) {}
};

/**
 * @brief Get threshold candidates (midpoints between sorted unique values).
 *
 * @param data The input data matrix (Eigen::MatrixXd)
 * @param indices The indices of the samples to consider for this feature
 * @param feature_index The index of the feature to consider for splitting
 *
 * @return std::vector<double> A vector of threshold candidates
 */
std::vector<double> get_threshold_candidates(
    const Eigen::Ref<const Eigen::MatrixXd> &data,
    const std::vector<int> &indices,
    int feature_index) noexcept
{
    // Extract feature values for the specified indices
    Eigen::VectorXd values(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        values(i) = data(indices[i], feature_index);
    }

    // Convert to std::vector for sorting (Eigen doesn't have direct sorting)
    std::vector<double> sorted_values(values.data(), values.data() + values.size());
    std::sort(sorted_values.begin(), sorted_values.end());

    // Find unique values and compute midpoints between them
    std::vector<double> thresholds;
    thresholds.reserve(sorted_values.size() / 2);
    for (size_t i = 0; i < sorted_values.size() - 1; ++i)
    {
        if (sorted_values[i] != sorted_values[i + 1])
        {
            thresholds.push_back((sorted_values[i] + sorted_values[i + 1]) / 2.0);
        }
    }
    return thresholds;
}

/**
 * @brief Partition sample indices based on a given feature and threshold. This
 * effectively splits the dataset into two children nodes.
 *
 * @param data The input data matrix (Eigen::MatrixXd)
 * @param indices The indices of the samples to consider for this feature
 * @param feature_index The index of the feature to consider for splitting
 * @param threshold The threshold value for the split
 * @param left_indices The output vector for left indices
 * @param right_indices The output vector for right indices
 */
void partition_indices(
    const Eigen::Ref<const Eigen::MatrixXd> &data,
    const std::vector<int> &indices,
    int feature_index, double threshold,
    std::vector<int> &left_indices,
    std::vector<int> &right_indices) noexcept
{
    left_indices.clear();
    right_indices.clear();
    left_indices.reserve(indices.size());
    right_indices.reserve(indices.size());

    for (int idx : indices)
    {
        if (data(idx, feature_index) <= threshold)
        {
            left_indices.push_back(idx);
        }
        else
        {
            right_indices.push_back(idx);
        }
    }
}

/**
 * @brief Compute class counts from indices.
 *
 * @param labels The labels vector (Eigen::VectorXi)
 * @param indices The indices of the samples to consider
 * @param n_classes The number of classes
 *
 * @return Eigen::VectorXi A vector of class counts
 */
Eigen::VectorXi compute_class_counts(
    const Eigen::Ref<const Eigen::VectorXi> &labels,
    const std::vector<int> &indices,
    int n_classes) noexcept
{
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(n_classes);
    for (int idx : indices)
    {
        counts(labels(idx))++;
    }
    return counts;
}

/**
 * @class A Decision Tree Classifier using Eigen for efficient matrix operations
 */
class DecisionTreeClassifier
{
public:
    // Constructor
    DecisionTreeClassifier(
        int max_depth = DEFAULT_MAX_DEPTH,
        int min_samples_split = DEFAULT_MIN_SAMPLES_SPLIT,
        const std::string_view criterion = "gini",
        double min_impurity_decrease = DEFAULT_MIN_IMPURITY_DECREASE)
        : max_depth_(max_depth),
          min_samples_split_(min_samples_split),
          min_impurity_decrease_(min_impurity_decrease),
          n_classes_(0),
          X_fitted_(0, 0), // Initialize with empty matrices
          y_fitted_(0)
    {
        // Convert string to enum for better performance
        if (criterion == "gini")
        {
            criterion_ = Criterion::Gini;
        }
        else if (criterion == "entropy")
        {
            criterion_ = Criterion::Entropy;
        }
        else
        {
            throw std::invalid_argument("criterion must be 'gini' or 'entropy'");
        }
    }

    /**
     * @brief Fit the tree on data X and labels y (Eigen version for direct NumPy arrays).
     *
     * @param X The input data matrix (Eigen::MatrixXd)
     * @param y The labels vector (Eigen::VectorXi)
     */
    void fit(const Eigen::Ref<const Eigen::MatrixXd> &X, const Eigen::Ref<const Eigen::VectorXi> &y)
    {
        if (X.rows() == 0)
        {
            throw std::runtime_error("Input X is empty");
        }
        if (X.rows() != y.size())
        {
            throw std::runtime_error("X and y must have the same number of samples");
        }

        // Keep a copy of the data internally (shallow copy references from NumPy)
        X_fitted_ = X;
        y_fitted_ = y;

        // Determine the number of classes
        n_classes_ = 0;
        for (int i = 0; i < y.size(); ++i)
        {
            int label = y(i);
            if (label < 0)
            {
                throw std::runtime_error("Labels must be non-negative integers (0..C-1)");
            }
            if (label >= n_classes_)
            {
                n_classes_ = label + 1;
            }
        }

        // Create initial indices for the entire dataset
        std::vector<int> indices(X.rows());
        std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...

        // Build the tree recursively
        root_ = build_tree(indices, 0);
    }

    /**
     * @brief Overload for Python list inputs (std::vector<std::vector<double>>).
     *
     * @param X The input data matrix (std::vector<std::vector<double>>)
     * @param y The labels vector (std::vector<int>)
     */
    void fit(const std::vector<std::vector<double>> &X, const std::vector<int> &y)
    {
        // Convert to Eigen types
        if (X.empty() || y.empty())
        {
            throw std::runtime_error("Empty input data");
        }

        const size_t n_samples = X.size();
        const size_t n_features = X[0].size();

        Eigen::MatrixXd X_eigen(n_samples, n_features);
        Eigen::VectorXi y_eigen(n_samples);

        for (size_t i = 0; i < n_samples; ++i)
        {
            if (X[i].size() != n_features)
            {
                throw std::runtime_error("All rows in X must have the same number of features");
            }

            for (size_t j = 0; j < n_features; ++j)
            {
                X_eigen(i, j) = X[i][j];
            }
            y_eigen(i) = y[i];
        }

        // Call the Eigen fit member function
        fit(X_eigen, y_eigen);
    }

    /**
     * @brief Predict using Eigen for efficient matrix operations. See
     * https://www.ibm.com/docs/en/zos/2.4.0?topic=pdpp-pragma-omp-parallel
     * for more details.
     *
     * @param X The input data matrix (Eigen::MatrixXd) for prediction
     *
     * @return Eigen::VectorXi A vector of predicted labels
     */
    Eigen::VectorXi predict(const Eigen::Ref<const Eigen::MatrixXd> &X) const
    {
        Eigen::VectorXi predictions(X.rows());

#pragma omp parallel for if (X.rows() > 10000)
        for (int i = 0; i < X.rows(); ++i)
        {
            predictions(i) = predict_single(X.row(i));
        }

        return predictions;
    }

    /**
     * @brief Overload for Python list compatibility (std::vector<std::vector<double>>).
     *
     * @param X The input data matrix (std::vector<std::vector<double>>) for prediction
     *
     * @return std::vector<int> A vector of predicted labels
     */
    std::vector<int> predict(const std::vector<std::vector<double>> &X) const
    {
        if (X.empty())
        {
            return {};
        }

        // Convert to Eigen
        const size_t n_samples = X.size();
        const size_t n_features = X[0].size();

        Eigen::MatrixXd X_eigen(n_samples, n_features);

        for (size_t i = 0; i < n_samples; ++i)
        {
            if (X[i].size() != n_features)
            {
                throw std::runtime_error("All rows in X must have the same number of features");
            }

            for (size_t j = 0; j < n_features; ++j)
            {
                X_eigen(i, j) = X[i][j];
            }
        }

        // Get predictions as Eigen vector
        Eigen::VectorXi eigen_preds = predict(X_eigen);

        // Convert back to std::vector
        std::vector<int> predictions(eigen_preds.data(), eigen_preds.data() + eigen_preds.size());
        return predictions;
    }

private:
    std::unique_ptr<TreeNode> root_; // Smart pointer to the root node of the tree
    int max_depth_;                  // Maximum depth of the tree
    int min_samples_split_;          // Minimum samples required to split an internal node
    int n_classes_;                  // Number of classes in the dataset
    Criterion criterion_;            // Enum for better performance
    double min_impurity_decrease_;   // Minimum impurity decrease required for splitting

    // Store references to the training data
    Eigen::MatrixXd X_fitted_;
    Eigen::VectorXi y_fitted_;

    /**
     * @brief Find majority class efficiently using Eigen
     *
     * @param counts The class counts vector (Eigen::VectorXi)
     *
     * @return int The index of the majority class
     */
    int find_majority_class(const Eigen::VectorXi &counts) const noexcept
    {
        int max_count = counts(0);
        int max_idx = 0;

        for (int i = 1; i < counts.size(); ++i)
        {
            if (counts(i) > max_count)
            {
                max_count = counts(i);
                max_idx = i;
            }
        }

        return max_idx;
    }

    /**
     * @brief Create a leaf node with the majority class.
     *
     * @param counts The class counts vector (Eigen::VectorXi)
     *
     * @return std::unique_ptr<TreeNode> A smart pointer to the created leaf node
     */
    std::unique_ptr<TreeNode> create_leaf_node(const Eigen::VectorXi &counts) noexcept
    {
        auto leaf = std::make_unique<TreeNode>();
        leaf->is_leaf = true;
        leaf->label = find_majority_class(counts);
        return leaf;
    }

    /**
     * @brief Recursively build the decision tree and return a smart pointer to the root node.
     *
     * @param indices The indices of the samples to consider for this node
     * @param depth The current depth of the tree
     *
     * @return std::unique_ptr<TreeNode> A smart pointer to the root node of the tree
     */
    std::unique_ptr<TreeNode> build_tree(
        const std::vector<int> &indices,
        int depth)
    {
        if (indices.empty())
        {
            return nullptr;
        }

        // Compute class counts for the current node
        Eigen::VectorXi current_counts = compute_class_counts(y_fitted_, indices, n_classes_);

        // Check stopping conditions: all labels the same, maximum depth reached, or insufficient samples
        bool all_same_label = true;
        int first_label = -1;

        for (int i = 0; i < n_classes_; ++i)
        {
            if (current_counts(i) > 0)
            {
                if (first_label == -1)
                {
                    first_label = i;
                }
                else if (i != first_label)
                {
                    all_same_label = false;
                    break;
                }
            }
        }

        // If any stopping condition is met, create a leaf node
        if (all_same_label || depth >= max_depth_ || static_cast<int>(indices.size()) < min_samples_split_)
        {
            return create_leaf_node(current_counts);
        }

        // Variables to store the best split
        int best_feature = -1;
        double best_threshold = 0.0;
        double best_impurity = std::numeric_limits<double>::infinity();
        double current_impurity = compute_impurity_from_counts(current_counts, indices.size());

        std::vector<int> best_left_indices, best_right_indices;

        const int n_features = X_fitted_.cols();

        // Evaluate each feature as a candidate for splitting
        for (int feature_index = 0; feature_index < n_features; ++feature_index)
        {
            // Get threshold candidates (midpoints between sorted unique values)
            std::vector<double> thresholds = get_threshold_candidates(X_fitted_, indices, feature_index);

            // Try each threshold
            for (double threshold : thresholds)
            {
                std::vector<int> left_indices, right_indices;
                partition_indices(X_fitted_, indices, feature_index, threshold, left_indices, right_indices);

                // Skip if one side is empty
                if (left_indices.empty() || right_indices.empty())
                {
                    continue;
                }

                // Compute class counts for left and right partitions
                Eigen::VectorXi left_counts = compute_class_counts(y_fitted_, left_indices, n_classes_);
                Eigen::VectorXi right_counts = compute_class_counts(y_fitted_, right_indices, n_classes_);

                // Compute impurity for left and right partitions
                double impurity_left = compute_impurity_from_counts(left_counts, left_indices.size());
                double impurity_right = compute_impurity_from_counts(right_counts, right_indices.size());

                // Weighted average of left and right impurities
                double weight_left = static_cast<double>(left_indices.size()) / indices.size();
                double weight_right = static_cast<double>(right_indices.size()) / indices.size();
                double weighted_impurity = weight_left * impurity_left + weight_right * impurity_right;

                // Impurity gain
                double impurity_gain = current_impurity - weighted_impurity;

                // Update best split if this one is better
                if (impurity_gain > min_impurity_decrease_ && weighted_impurity < best_impurity)
                {
                    best_impurity = weighted_impurity;
                    best_feature = feature_index;
                    best_threshold = threshold;
                    best_left_indices = std::move(left_indices);
                    best_right_indices = std::move(right_indices);
                }
            }
        }

        // If no valid split was found, create a leaf node
        if (best_feature == -1)
        {
            return create_leaf_node(current_counts);
        }

        // Create an internal node with the best split
        auto node = std::make_unique<TreeNode>();
        node->feature_index = best_feature;
        node->threshold = best_threshold;
        node->is_leaf = false;
        node->left = build_tree(best_left_indices, depth + 1);
        node->right = build_tree(best_right_indices, depth + 1);

        return node;
    }

    /**
     * @brief Predict the label for a single sample.
     *
     * @param row The input sample (Eigen::RowVectorXd)
     *
     * @return int The predicted label
     */
    int predict_single(const Eigen::Ref<const Eigen::RowVectorXd> &row) const noexcept
    {
        // Get the raw pointer to the root node
        TreeNode *node = root_.get();
        while (node && !node->is_leaf)
        {
            if (node->feature_index < row.size())
            {
                node = (row(node->feature_index) <= node->threshold) ? node->left.get() : node->right.get();
            }
            else
            {
                // Handle potential out-of-bounds index gracefully
                break;
            }
        }
        return (node) ? node->label : -1;
    }

    /**
     * @brief Compute impurity from pre-computed class counts (Gini or Entropy).
     *
     * @param counts The class counts vector (Eigen::VectorXi)
     * @param total The total number of samples
     *
     * @return double The computed impurity value
     */
    double compute_impurity_from_counts(const Eigen::VectorXi &counts, size_t total) const noexcept
    {
        if (total == 0)
            return 0.0;

        double m = static_cast<double>(total);

        if (criterion_ == Criterion::Gini)
        {
            // Gini impurity: 1 - sum(p_i^2)
            double sum_sq = 0.0;
            for (int i = 0; i < counts.size(); ++i)
            {
                double p = counts(i) / m;
                sum_sq += p * p;
            }
            return 1.0 - sum_sq;
        }
        else // Criterion::Entropy
        {
            // Entropy: - sum(p_i * log2(p_i))
            double entropy = 0.0;
            for (int i = 0; i < counts.size(); ++i)
            {
                if (counts(i) > 0)
                {
                    double p = counts(i) / m;
                    entropy -= p * std::log2(p);
                }
            }
            return entropy;
        }
    }
};

//
// Pybind11 module
//
PYBIND11_MODULE(_dtree, m)
{
    // Make the enum available to Python
    py::enum_<Criterion>(m, "Criterion")
        .value("Gini", Criterion::Gini)
        .value("Entropy", Criterion::Entropy)
        .export_values();

    py::class_<DecisionTreeClassifier>(m, "DecisionTreeClassifier")
        .def(
            py::init<int, int, const std::string_view, double>(),
            py::arg("max_depth") = DEFAULT_MAX_DEPTH,
            py::arg("min_samples_split") = DEFAULT_MIN_SAMPLES_SPLIT,
            py::arg("criterion") = "gini",
            py::arg("min_impurity_decrease") = DEFAULT_MIN_IMPURITY_DECREASE,
            R"pbdoc(
                Decision Tree Classifier

                Parameters
                ----------
                max_depth : int, default=3
                    The maximum depth of the tree. If None, nodes are expanded until
                    all leaves are pure or contain less than min_samples_split samples.
                
                min_samples_split : int, default=2
                    The minimum number of samples required to split an internal node.
                
                criterion : {"gini", "entropy"}, default="gini"
                    The function to measure the quality of a split.
                
                min_impurity_decrease : float, default=1e-7
                    A node will be split if this split induces a decrease of the impurity
                    greater than or equal to this value.
            )pbdoc")
        // Support both Eigen and list methods with proper docstrings
        .def("fit",
             static_cast<void (DecisionTreeClassifier::*)(const Eigen::Ref<const Eigen::MatrixXd> &, const Eigen::Ref<const Eigen::VectorXi> &)>(&DecisionTreeClassifier::fit),
             py::arg("X"), py::arg("y"),
             R"pbdoc(
                Build a decision tree classifier from the training set (X, y).
                This version accepts numpy arrays directly with zero-copy.

                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    The training input samples.
                y : array-like of shape (n_samples,)
                    The target values as integers.
            )pbdoc")
        .def("fit",
             static_cast<void (DecisionTreeClassifier::*)(const std::vector<std::vector<double>> &, const std::vector<int> &)>(&DecisionTreeClassifier::fit),
             py::arg("X"), py::arg("y"),
             R"pbdoc(
                Build a decision tree classifier from the training set (X, y).
                This version accepts Python lists.

                Parameters
                ----------
                X : list of lists of float
                    The training input samples.
                y : list of int
                    The target values as integers.
            )pbdoc")
        .def("predict",
             static_cast<Eigen::VectorXi (DecisionTreeClassifier::*)(const Eigen::Ref<const Eigen::MatrixXd> &) const>(&DecisionTreeClassifier::predict),
             py::arg("X"),
             R"pbdoc(
                Predict class values for X.
                This version accepts numpy arrays directly with zero-copy.

                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    The input samples.
                
                Returns
                -------
                y : ndarray of shape (n_samples,)
                    The predicted classes.
            )pbdoc")
        .def("predict",
             static_cast<std::vector<int> (DecisionTreeClassifier::*)(const std::vector<std::vector<double>> &) const>(&DecisionTreeClassifier::predict),
             py::arg("X"),
             R"pbdoc(
                Predict class values for X.
                This version accepts Python lists.

                Parameters
                ----------
                X : list of lists of float
                    The input samples.
                
                Returns
                -------
                y : list of int
                    The predicted classes.
            )pbdoc");

    // Module docstring
    m.doc() = R"pbdoc(
        Decision Tree Classifier module
        
        This module provides a fast C++ implementation of Decision Trees
        using Eigen for efficient numerical operations.
    )pbdoc";
}
