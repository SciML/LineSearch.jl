using SciMLTesting, LineSearch

# ExplicitImports only sees an extension module once its trigger package is loaded, so
# load the weakdeps here to bring LineSearchLineSearchesExt into the QA scan.
using LineSearches

run_qa(LineSearch)
