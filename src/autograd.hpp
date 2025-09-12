#include "node.hpp"
#include "ops_def.hpp"
#include "scalar_ops.hpp"
#include "tensor_ops.hpp"
#include <functional>
#include <type_traits>
#include <memory>
#include <stack>
#include <stdexcept>
#pragma once
// ================== 包装类 Value ==================
template<typename nT> class AutoGradGraph;

#define CHECK_GRAPH_SAME(ga, gb) do{if((ga) != (gb)) throw std::runtime_error("node's graph not same");}while(0);

template<typename T>
class Value {
private:
    Node<T>* node;
    AutoGradGraph<T>* graph;
public:
    friend class AutoGradGraph<T>;
    Value(Node<T>* n, AutoGradGraph<T>* g) : node(n), graph(g) {}
    T value() const { return node->value; }
    T& value() { return node->value; }
    T grad() const { return node->grad; }
    Node<T>* get_node()  const { return node; }
    AutoGradGraph<T>* get_graph()  const { return graph; }
};
// ================== 图执行 ==================
template<typename nT>
class AutoGradGraph {
    std::vector<Node<nT>*> topo;
    std::vector<std::unique_ptr<Node<nT>>> nodes;
    Node<nT>* output_node = nullptr;

    void _build(Node<nT>* output){
        topo.clear();
        topo.reserve(nodes.size() + 1);
        output_node = nullptr;
        std::vector<bool> vis(nodes.size(),false);
        std::stack<std::pair<Node<nT>*, bool>> st;
        st.emplace(output, false);
        while (!st.empty()) {
            auto [u, expanded] = st.top();
            st.pop();

            if (vis[u->id] && !expanded) {
                continue;
            }
            if (expanded) {
                topo.push_back(u);
            }else{
                vis[u->id] = true;
                st.emplace(u, true);
                for (size_t i = 0; i < u->input_size; i++) {
                    Node<nT>* v = u->inputs[i];
                    if (v != nullptr && !vis[v->id]) {
                        st.emplace(v, false);
                    }
                }
            }
        }

        this->output_node = output;
    }
public:
    template <class T, class... Args>
    T* add_node(Args&&... args) {
        nodes.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
        Node<nT>* node = nodes.back().get();
        node->id = nodes.size() - 1;
        return static_cast<T*>(node);
    }
    void build(){
        if(!nodes.empty())
            _build(nodes.back().get());
    }
    void build(Value<nT>& out){
        if(out.graph == this){
            _build(out.node);
        }
    }
    void zero_grad() { 
        for (auto* n : topo){
            n->zero_grad(); 
        }
    }
    void forward() {
        if(this->output_node == nullptr){
            build();
        } 
        for (auto* n : topo){
            n->forward(); 
        }
    }
    void backward(nT&& output_grad) {
        if(this->output_node == nullptr){
            build();
        }
        zero_grad();
        output_node->grad = output_grad;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->backward();
        }
    }

    Value<nT> var(nT&& v, bool require_grad = true) { 
        return Value<nT>(this->add_node<Variable<nT>>(v,require_grad),this); 
    }

    Value<nT> pow(const Value<nT>& a, double n) { 
        CHECK_GRAPH_SAME(a.graph, this);
        return Value<nT>(this->add_node<Pow<nT>>(a.node, n),this); 
    }
    Value<nT> sin(const Value<nT>& a) {
        CHECK_GRAPH_SAME(a.graph, this);
        return Value<nT>(this->add_node<UnaryFunc<nT,SinOp>>(a.node),this); 
    }
    Value<nT> cos(const Value<nT>& a) { 
        CHECK_GRAPH_SAME(a.graph, this);
        return Value<nT>(this->add_node<UnaryFunc<nT,CosOp>>(a.node),this);
    }
    Value<nT> exp(const Value<nT>& a) {
        CHECK_GRAPH_SAME(a.graph, this);
        return Value<nT>(this->add_node<UnaryFunc<nT,ExpOp>>(a.node),this); 
    }
    Value<nT> log(const Value<nT>& a) {
        CHECK_GRAPH_SAME(a.graph, this);
        return Value<nT>(this->add_node<UnaryFunc<nT,LogOp>>(a.node),this); 
    }
    Value<nT> relu(const Value<nT>& a) {
        CHECK_GRAPH_SAME(a.graph, this);
        return Value<nT>(this->add_node<UnaryFunc<nT,ReLUOp>>(a.node),this); 
    }
    Value<nT> sigmoid(const Value<nT>& a) {
        CHECK_GRAPH_SAME(a.graph, this);
        return Value<nT>(this->add_node<UnaryFunc<nT,SigmoidOp>>(a.node),this); 
    }
    Value<nT> tanh(const Value<nT>& a) {
        CHECK_GRAPH_SAME(a.graph, this);
        return Value<nT>(this->add_node<UnaryFunc<nT,TanhOp>>(a.node),this); 
    }
    //Tensor only ops
    Value<nT> max(const Value<nT>& a) {
        CHECK_GRAPH_SAME(a.graph, this);
        static_assert(std::is_same_v<nT, Tensor> == true);
        return Value<nT>(this->add_node<TensorOps::Reduce<ReduceType::Max>>(a.node),this); 
    }
    Value<nT> min(const Value<nT>& a) {
        CHECK_GRAPH_SAME(a.graph, this);
        static_assert(std::is_same_v<nT, Tensor> == true);
        return Value<nT>(this->add_node<TensorOps::Reduce<ReduceType::Min>>(a.node),this); 
    }
    Value<nT> sum(const Value<nT>& a) {
        CHECK_GRAPH_SAME(a.graph, this);
        static_assert(std::is_same_v<nT, Tensor> == true);
        return Value<nT>(this->add_node<TensorOps::Reduce<ReduceType::Sum>>(a.node),this); 
    }
    Value<nT> conv2d(const Value<nT>& a, const Value<nT>& k,
                size_t stride = 1, size_t pad = 0, size_t dilation = 1) {
        CHECK_GRAPH_SAME(a.graph, this);
        static_assert(std::is_same_v<nT, Tensor> == true);
        return Value<nT>(this->add_node<TensorOps::Conv2d>(a.node, k.node, stride, pad ,dilation),this); 
    }
    Value<nT> matmul(const Value<nT>& a, const Value<nT>& b) {
        CHECK_GRAPH_SAME(a.graph, this);
        static_assert(std::is_same_v<nT, Tensor> == true);
        return Value<nT>(this->add_node<TensorOps::MatMul>(a.node, b.node),this); 
    }
    Value<nT> reshape(const Value<nT>& x, std::vector<size_t>&& shape) {
        CHECK_GRAPH_SAME(x.graph, this);
        static_assert(std::is_same_v<nT, Tensor> == true);
        return Value<nT>(this->add_node<TensorOps::Reshape>(x.node, shape),this); 
    }
    Value<nT> maxpool2d(const Value<nT>& x, size_t kH, size_t kW,
            size_t stride = 1, size_t pad = 0) {
        CHECK_GRAPH_SAME(x.graph, this);
        static_assert(std::is_same_v<nT, Tensor> == true);
        return Value<nT>(this->add_node<TensorOps::MaxPool2d>(x.node, kH, kW, stride, pad),this); 
    }
};
template<typename nT>
Value<nT> operator+(const Value<nT>& a,const Value<nT>& b){
    CHECK_GRAPH_SAME(a.get_graph(), b.get_graph());
    return Value<nT>(a.get_graph()->template add_node<Add<nT>>(a.get_node(),b.get_node()),a.get_graph()); 
}

template<typename nT>
Value<nT> operator-(const Value<nT>& a,const Value<nT>& b){
    CHECK_GRAPH_SAME(a.get_graph(), b.get_graph());
    return Value<nT>(a.get_graph()->template add_node<Sub<nT>>(a.get_node(),b.get_node()),a.get_graph()); 
}

template<typename nT>
Value<nT> operator*(const Value<nT>& a,const Value<nT>& b){
    CHECK_GRAPH_SAME(a.get_graph(), b.get_graph());
    return Value<nT>(a.get_graph()->template add_node<Mul<nT>>(a.get_node(),b.get_node()),a.get_graph()); 
}

template<typename nT>
Value<nT> operator/(const Value<nT>& a,const Value<nT>& b){ 
    CHECK_GRAPH_SAME(a.get_graph(), b.get_graph());
    return Value<nT>(a.get_graph()->template add_node<Div<nT>>(a.get_node(),b.get_node()),a.get_graph()); 
}

