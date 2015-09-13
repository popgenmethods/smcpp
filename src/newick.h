#ifndef NEWICK_H
#define NEWICK_H

#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <queue>
#include <vector>
#include <string>
#include <iostream>

namespace parser
{
    // Forward declaration for the vector
    struct ptree;

    // typedef to ease the writing
    typedef std::vector<ptree> children_vector;

    // The tree structure itseflf
    struct ptree
    {
        children_vector children;
        std::string name;
        double length;
    };

    // Streaming operator for printing the result
    std::ostream& operator<<(std::ostream& stream, const ptree& tree)
    {
        bool first = true;
        stream << "(" << tree.name << ": " << tree.length << " { ";
        for (auto child: tree.children)
        {
            stream << (first ? "" : "," ) << child;
            first = false;
        }

        stream << " }";
        return stream;
    }
}

// adapt the structure to fusion phoenix
BOOST_FUSION_ADAPT_STRUCT(
    parser::ptree,
    (parser::children_vector, children)
    (std::string, name)
    (double, length)
)

namespace parser
{
    // namespace aliasing to shorten the names
    namespace qi = boost::spirit::qi;    
    namespace phoenix = boost::phoenix;

    // This grammar parse string to a ptree
    struct newick_grammar : qi::grammar<std::string::const_iterator, ptree()>
    {
    public:
        newick_grammar() 
            : newick_grammar::base_type(tree) // We try to parse the tree rule
        {                
            using phoenix::at_c; // Access nth field of structure
            using phoenix::push_back; // Push into vector

            // For label use %= to assign the result of the parse to the string
            label %= qi::lexeme[+(qi::char_ - ':' - ')' - ',')]; 

            // For branch length use %= to assign the result of the parse to the
            // double
            branch_length %= ':' >> qi::double_;

            // When parsing the subtree just assign the elements that have been
            // built in the subrules
            subtree = 
                // Assign vector of children to the first element of the struct
                -descendant_list [at_c<0>(qi::_val) = qi::_1 ] 
                // Assign the label to the second element
                >> -label [ at_c<1>(qi::_val) = qi::_1 ]
                // Assign the branch length to the third element 
                >> -branch_length [ at_c<2>(qi::_val) = qi::_1 ];

            // Descendant list is a vector of ptree, we just push back the
            // created ptrees into the vector
            descendant_list = 
                '(' >> subtree [ push_back(qi::_val, qi::_1) ]
                >> *(',' >> subtree [ push_back(qi::_val, qi::_1) ])
                >> ')';

            // The tree receive the whole subtree using %=
            tree %= subtree  >> ';' ;
        }

    private:
        // Here are the various grammar rules typed by the element they do
        // generate
        qi::rule<std::string::const_iterator, ptree()> tree, subtree;
        qi::rule<std::string::const_iterator, children_vector()> descendant_list;
        qi::rule<std::string::const_iterator, double()> branch_length;
        qi::rule<std::string::const_iterator, std::string()> label;
    };
}

int nsubtend (const parser::ptree *root, const std::string &l1, const std::string &l2)
{
    if (root->name == l1 or root->name == l2) return 1;
    if (root->children.size() == 0) return 0;
    return nsubtend(&root->children[0], l1, l2) + nsubtend(&root->children[1], l1, l2);
}

const parser::ptree* mrca(const parser::ptree *root, const std::string &l1, const std::string &l2)
{
    int nl = nsubtend(&root->children[0], l1, l2), nr = nsubtend(&root->children[1], l1, l2);
    if (nl == 1)
        return root;
    return (nl != 0) ? mrca(&root->children[0], l1, l2) : mrca(&root->children[1], l1, l2);
}

double tmrca(const parser::ptree *root, const std::string &l1, const std::string &l2)
{
    if (l1 == l2)
        return 0.0;
    if (root->name == l1 or root->name == l2)
        return root->length;
    if (root->children.size() == 0)
        return 0.0;
    double left = tmrca(&root->children[0], l1, l2);
    double right = tmrca(&root->children[1], l1, l2);
    // root is the mrca
    if (left != 0 and right != 0)
        return left + right;
    if (left == 0 and right == 0)
        return 0.0;
    return root->length + ((left != 0) ? left : right);
}

double cython_tmrca(const std::string newick, const std::string l1, const std::string l2)
{
    namespace qi = boost::spirit::qi;
    parser::newick_grammar grammar;
    parser::ptree tree;
    bool result = qi::phrase_parse(newick.cbegin(), newick.cend(), grammar, qi::space, tree);
    const parser::ptree *m = mrca(&tree, l1, l2);
    return tmrca(m, l1, l2);
}

/*
int main(int argc, char const *argv[])
{
    namespace qi = boost::spirit::qi;
    std::string str(argv[1]), l1(argv[2]), l2(argv[3]);

    // Instantiate grammar and tree
    parser::newick_grammar grammar;
    parser::ptree tree;

    // Parse
    bool result = qi::phrase_parse(str.cbegin(), str.cend(), grammar, qi::space, tree);
    std::cout << tmrca(&tree, l1, l2) << std::endl;
    return 0;
}

*/
#endif
