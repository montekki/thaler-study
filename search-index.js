var searchIndex = JSON.parse('{\
"matrix_multiplication":{"doc":"","t":[3,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11],"n":["G","borrow","borrow_mut","clone","clone_into","evaluate","from","into","new","num_vars","to_evaluations","to_owned","to_univariate_at_point","try_from","try_into","type_id","vzip"],"q":["matrix_multiplication","","","","","","","","","","","","","","","",""],"d":["A polynomial of form $g(z) = \\\\tilde{f}_A(r_1,z) \\\\cdot …","","","","","","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","Create $g$ for evaluating $f_A \\\\cdot f_B$ at any given …","","","","","","","",""],"i":[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"f":[null,[[["",0]],["",0]],[[["",0]],["",0]],[[["g",3,[["",26,[["clone",8],["field",8]]]]]],["g",3,[["",26,[["clone",8],["field",8]]]]]],[[["",0],["",0]]],[[["g",3,[["fftfield",8]]]],["option",4,[["fftfield",8]]]],[[]],[[]],[[["usize",0]],["g",3,[["field",8]]]],[[["g",3,[["fftfield",8]]]],["usize",0]],[[["g",3,[["fftfield",8]]]],["vec",3,[["fftfield",8]]]],[[["",0]]],[[["g",3,[["fftfield",8]]],["usize",0]],["option",4,[["sparsepolynomial",3,[["fftfield",8]]]]]],[[],["result",4]],[[],["result",4]],[[["",0]],["typeid",3]],[[]]],"p":[[3,"G"]]},\
"multilinear_extensions":{"doc":"","t":[5,5],"n":["cti_multilinear_from_evaluations","vsbw_multilinear_from_evaluations"],"q":["multilinear_extensions",""],"d":["Evaluate multilinear extension with an algorith from <code>VSBW13</code>","Evaluate multilinear extension of with an algorithm from …"],"i":[0,0],"f":[[[],["field",8]],[[],["field",8]]],"p":[]},\
"sum_check_protocol":{"doc":"","t":[3,4,13,13,3,13,8,3,4,11,11,11,11,11,11,11,11,11,11,11,10,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,10,11,11,10,11,10,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,12,12,12],"n":["BooleanHypercube","Error","FinalRound","JthRound","Prover","ProverClaimMismatch","SumCheckPolynomial","Verifier","VerifierRoundResult","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","c_1","evaluate","fmt","fmt","from","from","from","from","from","into","into","into","into","into","into_iter","new","new","new","next","num_vars","round","round","to_evaluations","to_string","to_univariate_at_point","try_from","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","type_id","vzip","vzip","vzip","vzip","vzip","0","1","0","0"],"q":["sum_check_protocol","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","sum_check_protocol::Error","","sum_check_protocol::VerifierRoundResult",""],"d":["A convenient way to iterate over $n$-dimentional boolean …","An error type of sum check protocol","On final round the verifier outputs <code>true</code> or <code>false</code> if it …","On $j$-th round the verifier outputs a random $r_j$ value","The state of the Prover.","","An abstraction over all types of polynomials that may be …","The state of the Verifier.","Values returned by Validator as a result of its run on …","","","","","","","","","","","Get the value $C_1$ that prover claims equal true answer.","Evaluates <code>self</code> at a given point","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","Create an $n$-dimentional <code>BooleanHypercube</code>","Create a new <code>Prover</code> state with the polynomial $g$.","Create the new state of the <code>Verifier</code>.","","Returns the number of variables in <code>self</code>","Perform $j$-th round of the <code>Prover</code> side of the prococol.","Perform the $j$-th round of the [<code>Verifier]</code> side of the …","Returns a list of evaluations over the domain, which is the","","Given an index of a variable <code>i</code>, and a point <code>at</code> in $F^n$ …","","","","","","","","","","","","","","","","","","","","","","","",""],"i":[0,0,1,1,0,2,0,0,0,3,4,5,1,2,3,4,5,1,2,4,6,2,2,3,4,5,1,2,3,4,5,1,2,3,3,4,5,3,6,4,5,6,2,6,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,7,7,8,9],"f":[null,null,null,null,null,null,null,null,null,[[["",0]],["",0]],[[["",0]],["",0]],[[["",0]],["",0]],[[["",0]],["",0]],[[["",0]],["",0]],[[["",0]],["",0]],[[["",0]],["",0]],[[["",0]],["",0]],[[["",0]],["",0]],[[["",0]],["",0]],[[["prover",3,[["field",8],["sumcheckpolynomial",8,[["field",8]]]]]],["field",8]],[[["",0]],["option",4]],[[["error",4],["formatter",3]],["result",6]],[[["error",4],["formatter",3]],["result",6]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[["u32",0]],["booleanhypercube",3,[["field",8]]]],[[["sumcheckpolynomial",8,[["field",8]]]],["prover",3,[["field",8],["sumcheckpolynomial",8,[["field",8]]]]]],[[["usize",0],["field",8],["sumcheckpolynomial",8,[["field",8]]]],["verifier",3,[["field",8],["sumcheckpolynomial",8,[["field",8]]]]]],[[["booleanhypercube",3,[["field",8]]]],["option",4]],[[["",0]],["usize",0]],[[["prover",3,[["field",8],["sumcheckpolynomial",8,[["field",8]]]]],["field",8],["usize",0]],["option",4,[["sparsepolynomial",3,[["field",8]]]]]],[[["verifier",3,[["field",8],["sumcheckpolynomial",8,[["field",8]]]]],["sparsepolynomial",3,[["field",8]]],["",0]],["result",4,[["verifierroundresult",4,[["field",8]]],["error",4]]]],[[["",0]],["vec",3]],[[["",0]],["string",3]],[[["",0],["usize",0]],["option",4,[["sparsepolynomial",3]]]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[],["result",4]],[[["",0]],["typeid",3]],[[["",0]],["typeid",3]],[[["",0]],["typeid",3]],[[["",0]],["typeid",3]],[[["",0]],["typeid",3]],[[]],[[]],[[]],[[]],[[]],null,null,null,null],"p":[[4,"VerifierRoundResult"],[4,"Error"],[3,"BooleanHypercube"],[3,"Prover"],[3,"Verifier"],[8,"SumCheckPolynomial"],[13,"ProverClaimMismatch"],[13,"JthRound"],[13,"FinalRound"]]},\
"thaler_study":{"doc":"","t":[],"n":[],"q":[],"d":[],"i":[],"f":[],"p":[]},\
"triangle_counting":{"doc":"","t":[3,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11],"n":["G","borrow","borrow_mut","clone","clone_into","evaluate","from","into","new_adj_matrix","num_vars","to_evaluations","to_owned","to_univariate_at_point","try_from","try_into","type_id","vzip"],"q":["triangle_counting","","","","","","","","","","","","","","","",""],"d":["","","","","","Evaluate over a point $(X, Y, Z)$.","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","","","","","","","","",""],"i":[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"f":[null,[[["",0]],["",0]],[[["",0]],["",0]],[[["g",3,[["",26,[["clone",8],["field",8]]]]]],["g",3,[["",26,[["clone",8],["field",8]]]]]],[[["",0],["",0]]],[[["g",3,[["fftfield",8]]]],["option",4,[["fftfield",8]]]],[[]],[[]],[[["usize",0]],["g",3,[["field",8]]]],[[["g",3,[["fftfield",8]]]],["usize",0]],[[["g",3,[["fftfield",8]]]],["vec",3,[["fftfield",8]]]],[[["",0]]],[[["g",3,[["fftfield",8]]],["usize",0]],["option",4,[["sparsepolynomial",3,[["fftfield",8]]]]]],[[],["result",4]],[[],["result",4]],[[["",0]],["typeid",3]],[[]]],"p":[[3,"G"]]}\
}');
if (typeof window !== 'undefined' && window.initSearch) {window.initSearch(searchIndex)};
if (typeof exports !== 'undefined') {exports.searchIndex = searchIndex};
