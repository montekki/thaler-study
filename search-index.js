var searchIndex = JSON.parse('{\
"fiat_shamir":{"doc":"Fiat-Shamir Transformation implementation.","t":"EDIIDGNNLLLLLLLLLLLLLLKFLLLKLKKLLLLLLLLLLLFLLL","n":["Error","FiatShamirTranscript","InteractiveProver","InteractiveVerifier","RandNums","Result","Serialization","SumCheck","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","draw","fmt","fmt","from","from","from","from","from","g_1","generate_transcript","into","into","into","num_rounds","provide","round","round","source","to_string","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","verify_transcript","vzip","vzip","vzip"],"q":["fiat_shamir","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],"d":["Crate error type.","A transcript of the Fiat-Shamir transformation.","A trait describing an Interactive Prover.","A trait describing an Interactive Verifier.","A helper struct to feed non-random values into interactive …","Crate <code>Result</code> type.","An error in ark_serialize.","A SumCheck error.","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","","Returns the argument unchanged.","","Get $g_1$.","Generate a Fiat-Shamir transcript turning an Interactive …","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Number of rounds.","","Perform a step with V’s challenge $r_i$.","Perform a round of the Interactive Verifier.","","","","","","","","","","","","Perform verification of the Fiat-Shamir transcript turning …","","",""],"i":[0,0,0,0,0,0,3,3,13,2,3,13,2,3,2,3,3,13,2,3,3,3,12,0,13,2,3,12,3,12,22,3,3,13,2,3,13,2,3,13,2,3,0,13,2,3],"f":[0,0,0,0,0,0,0,0,[[]],[[]],[[]],[[]],[[]],[[]],[[[2,[1]]],1],[[3,4],5],[[3,4],5],[[]],[[]],[6,3],[[]],[7,3],[[],[[10,[[9,[8]]]]]],[[[12,[11]]],[[10,[13]]]],[[]],[[]],[[]],[[],14],[15],[14,[[10,[[9,[8]]]]]],[14,[[10,[16]]]],[3,[[18,[17]]]],[[],19],[[],20],[[],20],[[],20],[[],20],[[],20],[[],20],[[],21],[[],21],[[],21],[[13,[22,[11,[2,[11]]]]],[[10,[16]]]],[[]],[[]],[[]]],"p":[[8,"Copy"],[3,"RandNums"],[4,"Error"],[3,"Formatter"],[6,"Result"],[4,"SerializationError"],[4,"Error"],[15,"u8"],[3,"Vec"],[6,"Result"],[8,"Field"],[8,"InteractiveProver"],[3,"FiatShamirTranscript"],[15,"usize"],[3,"Demand"],[15,"bool"],[8,"Error"],[4,"Option"],[3,"String"],[4,"Result"],[3,"TypeId"],[8,"InteractiveVerifier"]]},\
"gkr_protocol":{"doc":"The implementation of the GKR protocol.","t":"NENNDENGNNNNDENLLLLLLLLLLLLLLLLLLLLLLLLLLLLLFLLLLLFLLLLLLLLLLLLLLLLLLLLLLLLMMMMMMMMM","n":["Begin","Error","FinalRoundMessage","FirstRound","Prover","ProverMessage","R","Result","RoundStarted","StartSumCheck","SumCheckProverMessage","SumCheckRoundResult","Verifier","VerifierMessage","WrongVerifierState","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","c_1","check_input","eq","equivalent","final_random_point","fmt","fmt","fmt","fmt","from","from","from","from","from","into","into","into","into","into","line","new","new","provide","receive_prover_msg","receive_verifier_msg","restrict_poly","round_msg","start_protocol","start_round","to_string","try_from","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","type_id","vzip","vzip","vzip","vzip","vzip","c_1","circuit_outputs","num_vars","p","p","q","round","r","res"],"q":["gkr_protocol","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","gkr_protocol::ProverMessage","","","","","","","gkr_protocol::VerifierMessage",""],"d":["<code>Prover</code> begins the protocol by the claim about the outputs.","GKR protocol error type.","In the final the restriction polynomial $q$ is added.","The first round has completed.","The state of the Prover.","Messages emitted by the <code>Prover</code>.","Sends out the $r_i$ to be used by the <code>Prover</code>.","GKR protocol result type.","The j-th round has started.","Instruct the <code>Verifier</code> to start a Sum-Check protocol for …","A step of the current sum-check protocol.","A result of running a step in the current sum check …","The state of the Verifier.","Messages emitted by the <code>Verifier</code>.","Wrong state.","","","","","","","","","","","Get the $c_1$ of the current Sum-Check prover.","Perform the final check of the input.","","","Final random point in the Sum-Check protocol.","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Compute a line","Create a new <code>Verifier</code> with the claim of the <code>Prover</code>.","Create a new <code>Prover</code> state from a circuit and an evaluation.","","Receive a message from <code>Prover</code>.","Receive a message from the <code>Verifier</code>.","Restrict a polynomial to a given line.","Perform a step of the Sum-Check protocol and provide a …","At the start of the protocol $P$ sends a function $D: …","Create a Sum-Check prover for round $i$.","","","","","","","","","","","","","","","","","","","","","","A $c_1$ from Sum-Check protocol.","Claimed outputs","A number of variables.","A polynomial sent at each step of Sum-Check.","A polynomial sent at each step of Sum-Check.","Sends a univariate polynomial $q$ of degree at most k_…","At which round of GKR this Sum-Check is being started.","$r_i$","Result of a Sum-Check round."],"i":[7,0,7,8,0,0,8,0,8,7,7,8,0,0,10,3,2,10,8,7,3,2,10,8,7,2,3,7,7,3,10,10,8,7,3,2,10,8,7,3,2,10,8,7,0,3,2,10,3,2,0,2,2,2,10,3,2,10,8,7,3,2,10,8,7,3,2,10,8,7,3,2,10,8,7,21,22,21,23,24,24,21,25,26],"f":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[[2,[1]]],1],[[[3,[1]]],4],[[[7,[[0,[5,6]]]],7],4],[[],4],[[[3,[1]]],[[9,[[8,[1]]]]]],[[10,11],12],[[10,11],12],[[[8,[[0,[13,6]]]],11],12],[[[7,[[0,[13,6]]]],11],12],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[],[[15,[[14,[6]]]]]],0,0,[16],[[[3,[1]],[7,[1]]],[[9,[[8,[1]]]]]],[[[2,[1]],[8,[1]]]],[[],[[14,[6]]]],[[[2,[1]],17],[[7,[1]]]],[[[2,[1]]],[[7,[1]]]],[[[2,[1]],17],[[7,[1]]]],[[],18],[[],19],[[],19],[[],19],[[],19],[[],19],[[],19],[[],19],[[],19],[[],19],[[],19],[[],20],[[],20],[[],20],[[],20],[[],20],[[]],[[]],[[]],[[]],[[]],0,0,0,0,0,0,0,0,0],"p":[[8,"FftField"],[3,"Prover"],[3,"Verifier"],[15,"bool"],[8,"PartialEq"],[8,"Field"],[4,"ProverMessage"],[4,"VerifierMessage"],[6,"Result"],[4,"Error"],[3,"Formatter"],[6,"Result"],[8,"Debug"],[3,"SparsePolynomial"],[3,"Vec"],[3,"Demand"],[15,"usize"],[3,"String"],[4,"Result"],[3,"TypeId"],[13,"StartSumCheck"],[13,"Begin"],[13,"SumCheckProverMessage"],[13,"FinalRoundMessage"],[13,"R"],[13,"SumCheckRoundResult"]]},\
"matrix_multiplication":{"doc":"","t":"DLLLLLLLLLLLLLLLLL","n":["G","borrow","borrow_mut","clone","clone_into","evaluate","fix_variables","from","into","new","num_vars","to_evaluations","to_owned","to_univariate","try_from","try_into","type_id","vzip"],"q":["matrix_multiplication","","","","","","","","","","","","","","","","",""],"d":["A polynomial of form $g(z) = \\\\tilde{f}_A(r_1,z) \\\\cdot …","","","","","","","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","Create $g$ for evaluating $f_A \\\\cdot f_B$ at any given …","","","","","","","",""],"i":[0,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],"f":[0,[[]],[[]],[[[3,[[0,[1,2]]]]],[[3,[[0,[1,2]]]]]],[[]],[[[3,[4]]],[[5,[4]]]],[[[3,[4]]],[[3,[4]]]],[[]],[[]],[6,[[3,[2]]]],[[[3,[4]]],6],[[[3,[4]]],[[7,[4]]]],[[]],[[[3,[4]]],[[8,[4]]]],[[],9],[[],9],[[],10],[[]]],"p":[[8,"Clone"],[8,"Field"],[3,"G"],[8,"FftField"],[4,"Option"],[15,"usize"],[3,"Vec"],[3,"SparsePolynomial"],[4,"Result"],[3,"TypeId"]]},\
"multilinear_extensions":{"doc":"","t":"FF","n":["cti_multilinear_from_evaluations","vsbw_multilinear_from_evaluations"],"q":["multilinear_extensions",""],"d":["Evaluate multilinear extension with an algorith from <code>VSBW13</code>","Evaluate multilinear extension of with an algorithm from …"],"i":[0,0],"f":[[[],1],[[],1]],"p":[[8,"Field"]]},\
"relaxed_pcs":{"doc":"The implementation of the Relaxed PCS protocol.","t":"NNENINNDGNQDLKLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL","n":["ArkCryptoPrimitivesError","DegreeMismatch","Error","EvalMismatch","IF","NoProverPoly","PolyEvalDimMismatch","Prover","Result","ToBytesError","Values","Verifier","all_multidimentional_values","all_values","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","challenge","challenge_prover","commited_univariate","fmt","fmt","from","from","from","from","into","into","into","merkle_root","new","new","poly_restriction_to_line","provide","random_line","source","to_string","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","verify_prover_reply","vzip","vzip","vzip"],"q":["relaxed_pcs","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],"d":["","","Crate error type.","","Iterate over all possible values of a finite field.","","","Prover in the Relaxed PCS protocol.","Crate <code>Result</code> type.","","Type of the values.","The Verifier in the Relaxed PCS protocol.","Get all permutations of the values of the type.","Get all values of the type.","","","","","","","Challenge","Challenge the prover at some point.","Receive the commited univariate polynomial from Prover.","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Get the merkle root.","Create a new Verifier.","Create a new Prover.","Restrict to line.","","Generate a random line to challenge the <code>Prover</code>.","","","","","","","","","","","","Verify the prover’s reply.","","",""],"i":[11,11,0,11,0,11,11,0,0,11,3,0,3,3,9,6,11,9,6,11,6,9,9,11,11,9,6,11,11,9,6,11,6,9,6,6,11,9,11,11,9,6,11,9,6,11,9,6,11,9,9,6,11],"f":[0,0,0,0,0,0,0,0,0,0,0,0,[1,[[2,[2]]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[[6,[3,[4,[3]],5]],[2,[3]]],7],[[[9,[8,5]]],[[2,[8]]]],[[[9,[8,5]],[10,[8]]],7],[[11,12],13],[[11,12],13],[[]],[[]],[[]],[14,11],[[]],[[]],[[]],[[[6,[3,[4,[3]],5]]]],[[1,1],[[9,[8,5]]]],[[[4,[3]]],[[7,[[6,[3,[4,[3]],5]]]]]],[[[6,[3,[4,[3]],5]]],[[10,[3]]]],[15],[[[9,[8,5]]]],[11,[[17,[16]]]],[[],18],[[],19],[[],19],[[],19],[[],19],[[],19],[[],19],[[],20],[[],20],[[],20],[[[9,[8,5]],[21,[5]],8],7],[[]],[[]],[[]]],"p":[[15,"usize"],[3,"Vec"],[8,"IF"],[8,"MultilinearExtension"],[8,"Config"],[3,"Prover"],[6,"Result"],[8,"Field"],[3,"Verifier"],[3,"SparsePolynomial"],[4,"Error"],[3,"Formatter"],[6,"Result"],[6,"Error"],[3,"Demand"],[8,"Error"],[4,"Option"],[3,"String"],[4,"Result"],[3,"TypeId"],[3,"Path"]]},\
"sum_check_protocol":{"doc":"","t":"DENNNDNIIDELLLLLLLLLLLKKKLLLLLLLLLLLLLLLLLLKLLLLLKLKLLLLLLLLLLLLLLLLLLLL","n":["BooleanHypercube","Error","FinalRound","JthRound","NoPolySet","Prover","ProverClaimMismatch","RngF","SumCheckPolynomial","Verifier","VerifierRoundResult","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","c_1","draw","evaluate","fix_variables","fmt","fmt","fmt","from","from","from","from","from","into","into","into","into","into","into_iter","new","new","new","next","num_vars","num_vars","provide","round","round","set_c_1","to_evaluations","to_string","to_univariate","try_from","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","type_id","vzip","vzip","vzip","vzip","vzip"],"q":["sum_check_protocol","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],"d":["A convenient way to iterate over $n$-dimentional boolean …","An error type of sum check protocol","On final round the verifier outputs <code>true</code> or <code>false</code> if it …","On $j$-th round the verifier outputs a random $r_j$ value","","The state of the Prover.","","","An abstraction over all types of polynomials that may be …","The state of the Verifier.","Values returned by Validator as a result of its run on …","","","","","","","","","","","Get the value $C_1$ that prover claims equal true answer.","","Evaluates <code>self</code> at a given point","Reduce the number of variables in <code>Self</code> by fixing a …","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","Create an $n$-dimentional <code>BooleanHypercube</code>","Create a new <code>Prover</code> state with the polynomial $g$.","Create the new state of the <code>Verifier</code>.","","Returns the number of variables in <code>self</code>","","","Perform $j$-th round of the <code>Prover</code> side of the prococol.","Perform the $j$-th round of the <code>Verifier</code> side of the …","","Returns a list of evaluations over the domain, which is the","","Compute the $j$-th round of polynomial for sumcheck over …","","","","","","","","","","","","","","","","","","","",""],"i":[0,0,9,9,5,0,5,0,0,0,0,11,3,13,5,9,11,3,13,5,9,3,20,2,2,5,5,9,11,3,13,5,9,11,3,13,5,9,11,11,3,13,11,2,3,5,3,13,13,2,5,2,11,3,13,5,9,11,3,13,5,9,11,3,13,5,9,11,3,13,5,9],"f":[0,0,0,0,0,0,0,0,0,0,0,[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[[3,[1,[2,[1]]]]],1],[[]],[[],4],[[]],[[5,6],7],[[5,6],7],[[[9,[[0,[8,1]]]],6],7],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[10,[[11,[1]]]],[[[2,[1]]],[[3,[1,[2,[1]]]]]],[[12,[4,[[2,[1]]]]],[[13,[1,[2,[1]]]]]],[[[11,[1]]],4],[[],12],[[[3,[1,[2,[1]]]]],12],[14],[[[3,[1,[2,[1]]]],1,12],[[15,[1]]]],[[[13,[1,[2,[1]]]],[15,[1]]],[[16,[[9,[1]],5]]]],[[[13,[1,[2,[1]]]],1]],[[],17],[[],18],[[],15],[[],16],[[],16],[[],16],[[],16],[[],16],[[],16],[[],16],[[],16],[[],16],[[],16],[[],19],[[],19],[[],19],[[],19],[[],19],[[]],[[]],[[]],[[]],[[]]],"p":[[8,"Field"],[8,"SumCheckPolynomial"],[3,"Prover"],[4,"Option"],[4,"Error"],[3,"Formatter"],[6,"Result"],[8,"Debug"],[4,"VerifierRoundResult"],[15,"u32"],[3,"BooleanHypercube"],[15,"usize"],[3,"Verifier"],[3,"Demand"],[3,"SparsePolynomial"],[4,"Result"],[3,"Vec"],[3,"String"],[3,"TypeId"],[8,"RngF"]]},\
"triangle_counting":{"doc":"","t":"DLLLLLLLLLLLLLLLLL","n":["G","borrow","borrow_mut","clone","clone_into","evaluate","fix_variables","from","into","new_adj_matrix","num_vars","to_evaluations","to_owned","to_univariate","try_from","try_into","type_id","vzip"],"q":["triangle_counting","","","","","","","","","","","","","","","","",""],"d":["A polynomial","","","","","","","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","Creates a new $3 \\\\log n$-variate polynomial $g(X,Y,Z)$ from","","","","","","","",""],"i":[0,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],"f":[0,[[]],[[]],[[[3,[[0,[1,2]]]]],[[3,[[0,[1,2]]]]]],[[]],[[[3,[4]]],[[5,[4]]]],[[[3,[4]]],[[3,[4]]]],[[]],[[]],[6,[[3,[2]]]],[[[3,[4]]],6],[[[3,[4]]],[[7,[4]]]],[[]],[[[3,[4]]],[[8,[4]]]],[[],9],[[],9],[[],10],[[]]],"p":[[8,"Clone"],[8,"Field"],[3,"G"],[8,"FftField"],[4,"Option"],[15,"usize"],[3,"Vec"],[3,"SparsePolynomial"],[4,"Result"],[3,"TypeId"]]}\
}');
if (typeof window !== 'undefined' && window.initSearch) {window.initSearch(searchIndex)};
if (typeof exports !== 'undefined') {exports.searchIndex = searchIndex};
