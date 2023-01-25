# Root's child must be an intent or a token
# A root should not have a parent ,
# An intents child must be a slot or a token.
# child : split on space, 3 possibilities: [njjfnj, ] ,  actual word.


# regex replace on close brackets.

# if the word is a token, then node list till that point cannot be empty.
# the last word needs to be a bracket closed.
# When we

def preProcess(predFile):
    import re
    avg_extra_forward = 0
    avg_extra_backward = 0
    # with open("Old Files/predWordNetPostProcess.txt", "a") as m:
    with open("t5_k2/epoch81/epoch_108_post_process.txt", "w") as m:
        with open(predFile, "r") as p:
            pred = p.readlines()
            for indexSentence, sent in enumerate(pred):
                if not sent.startswith("[IN"):
                    sent = "[IN" + sent
                # sent = sent.replace("\n","")
                sent = sent.replace("[ ", "[")
                listof_tuple = re.findall("(\])([A-Z]|[a-z])", sent)
                for bracket, character in listof_tuple:
                    index = sent.index(bracket + character)
                    sent = sent[:index] + "] " + character + sent[index + 1:]
                if not sent.endswith("] "):
                    sent = sent + "] "
                import re
                char = "(\[)IN:[A-Z]|[a-z]"
                indices = [i.start() for i in re.finditer(char, sent)]
                for index in indices:
                    next_space_index = sent.find(" ", index + 3, len(sent))
                    word = sent[next_space_index + 1:]
                    if sent[next_space_index + 1:].startswith("[IN:"):
                        sent = sent[:next_space_index + 1] + "] " + word
                Slot_indices = [i.start() for i in re.finditer("(\[)[SL:]", sent)]
                # if len(Slot_indices) > 1:
                #     Slot_indices = Slot_indices[1:]
                for index in Slot_indices:
                    next_space_index = sent.find(" ", index + 3, len(sent))
                    second_space_index = sent.find(" ", next_space_index + 1, len(sent))
                    st = sent[second_space_index + 1:]
                    sent = sent[:next_space_index + 1] + "] " + st
                    if sent[next_space_index + 1:].startswith("[IN:"):
                        sent = sent[:next_space_index + 1] + "] " + sent[next_space_index + 1:]
                sent_arr = sent.split(" ")
                num_words = len(sent_arr)
                countForward = sent.count("[")
                countBackward = sent.count("]")
                new_sent = ""
                countForward += 1
                if countBackward < countForward:
                    idealWordIndex = -1
                    for wordIndex, word in enumerate(sent_arr):
                        if idealWordIndex == wordIndex and countBackward < countForward:
                            new_sent += word + "] "
                            countBackward += 1
                        else:
                            new_sent += word + " "
                        if word != "" and word[-1] == ":" and wordIndex + 1 < num_words and sent_arr[wordIndex + 1][
                            -1] != "]":
                            idealWordIndex = wordIndex + 1
                    while countBackward < countForward:
                        new_sent += "]"
                        countBackward += 1
                elif countBackward > countForward + 1:
                    removeWordIndex = -1
                    for wordIndex, word in enumerate(sent_arr):
                        if removeWordIndex == wordIndex and countBackward > countForward:
                            new_sent += word[:-1] + " "
                            countBackward -= 1
                        else:
                            new_sent += word + " "
                        if word == "]" and wordIndex + 1 < num_words and sent_arr[wordIndex + 1] == "]":
                            removeWordIndex = wordIndex + 1
                if new_sent != "":
                    print("NEW_SENT", new_sent)
                    new_sent = new_sent.split("\n")[0]
                    print("NEW_SENT", new_sent)
                    if not new_sent.endswith("] "):
                        m.write(new_sent + "]")
                    else:
                        m.write(new_sent)
                else:
                    m.write(sent)


def preProcessTest():
    sent = ":GET_EVENT [SL:CATEGORY_EVENT fireworks ] [SL:LOCATION [IN:GET_LOCATION [SL:LOCATION_MODIFIER [IN:GET_LOCATION_MODIFIER [SL:MODIFIER_LOCATION_MODIFIER [SL:LOCATION_MODIFIER_MODIFIER_MODIFIER_MODIFIER_MODIFIER_MODIFIER_MODIFIER_MODIF]]]]]]]"
    sent_arr = sent.split(" ")
    num_words = len(sent_arr)
    countForward = sent.count("[")
    countBackward = sent.count("]")
    new_sent = ""
    new_sent += "[IN"
    countForward += 1
    if countBackward < countForward:
        idealWordIndex = -1
        for wordIndex, word in enumerate(sent_arr):
            if idealWordIndex == wordIndex and countBackward < countForward:
                new_sent += word + "] "
                countBackward += 1
            else:
                new_sent += word + " "
            if word != "" and word[-1] == ":" and wordIndex + 1 < num_words and sent_arr[wordIndex + 1][-1] != "]":
                idealWordIndex = wordIndex + 1
        while countBackward < countForward:
            new_sent += "]"
            countBackward += 1
    elif countBackward > countForward + 1:
        removeWordIndex = -1
        for wordIndex, word in enumerate(sent_arr):
            if removeWordIndex == wordIndex and countBackward > countForward:
                new_sent += word[:-1] + " "
                countBackward -= 1
            else:
                new_sent += word + " "
            if word == "]" and wordIndex + 1 < num_words and sent_arr[wordIndex + 1] == "]":
                removeWordIndex = wordIndex + 1
    if new_sent != "[IN":
        print(new_sent)
    else:
        print("[IN" + sent + "\n")
    a = "[IN" + sent + "\n"
    print(a.find("\n"))





# ^]

preProcess("t5_k2/epoch81/outputs_108_epochs.txt")
# preProcessTest()
