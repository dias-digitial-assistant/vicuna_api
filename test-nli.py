import NLI

nli_obj = NLI.NLI_German()
print(nli_obj.check_nli(premise="Heute is Sonntag",hypothesis="Heute ist Montag"))
