def fold_instrument(instrument):
    fold_ranges = []
    if instrument == "piano":
        fold_ranges = [
            list(range(1, 13)) + list(range(56, 68)) + list(range(203, 227)) + list(range(287, 299)),  # Fold 1
            list(range(13, 31)) + list(range(43, 56)) + list(range(227, 251)),  # fold2
            list(range(31, 43)) + list(range(86, 100)) + list(range(131, 155)) + list(range(191, 203)),  # fold3
            list(range(68, 86)) + list(range(155, 167)) + list(range(179, 191)) + list(range(263, 275)),  # fold4
            list(range(100, 131)) + list(range(167, 179)) + list(range(251, 263)) + list(range(275, 287))  # fold5

        ]
    elif instrument == "acoustic_guitar":
        fold_ranges = [
            list(range(1, 37)) + list(range(73, 109)) + list(range(212, 224)),  # Fold 1
            list(range(37, 49)) + list(range(121, 133)) + list(range(200, 212)) + list(range(272, 296)) + list(
                range(392, 404)),  # fold2
            list(range(49, 61)) + list(range(109, 121)) + list(range(160, 188)) + list(range(248, 260)) + list(
                range(368, 392)),  # fold3
            list(range(61, 73)) + list(range(145, 160)) + list(range(188, 200)) + list(range(236, 248)) + list(
                range(332, 356)),  # fold4
            list(range(133, 145)) + list(range(260, 272)) + list(range(296, 332)) + list(range(356, 368))  # fold5
        ]

    elif instrument == "electric_guitar":

        fold_ranges = [
            list(range(1, 13)) + list(range(25, 37)) + list(range(173, 186)) + list(range(210, 222)) + list(
                range(270, 282)) + list(range(354, 366)),  # Fold 1
            list(range(13, 25)) + list(range(73, 97)) + list(range(137, 149)) + list(range(186, 198)) + list(
                range(246, 258)),  # fold2
            list(range(37, 49)) + list(range(97, 109)) + list(range(133, 137)) + list(range(161, 173)) + list(
                range(198, 210)) + list(range(234, 246)) + list(range(282, 294)),  # fold3
            list(range(49, 73)) + list(range(109, 133)) + list(range(222, 234)) + list(range(306, 318)) + list(
                range(330, 342)),  # fold4
            list(range(149, 161)) + list(range(258, 270)) + list(range(294, 306)) + list(range(318, 330)) + list(
                range(342, 354)) + list(range(366, 378))  # fold5
        ]
    else:
        fold_ranges = [
            list(range(1, 37)) + list(range(73, 109)) + list(range(212, 224)) + list(range(404, 440)) + list(
                range(707, 739)) + list(range(1029, 1066)),  # Fold 1
            list(range(37, 49)) + list(range(121, 133)) + list(range(200, 212)) + list(
                range(392, 404)) + list(range(739, 776)) + list(range(993, 1029)),  # fold2
            list(range(49, 61)) + list(range(160, 188)) + list(range(248, 260)) + list(
                range(368, 392)) + list(range(477, 514)) + list(range(575, 600)),  # fold3
            list(range(61, 73)) + list(range(145, 160)) + list(range(188, 200)) + list(
                range(332, 356)) + list(range(440, 477)) + list(range(550, 575)),  # fold4
            list(range(133, 145)) + list(range(260, 272)) + list(range(296, 332)) + list(range(514, 550)) + list(
                range(707, 744)),  # fold5
            list(range(109, 121)) + list(range(236, 248)) + list(range(514, 550)) + list(range(600, 635)),  # fold6
            list(range(272, 296)) + list(range(356, 368)) + list(range(635, 671)) + list(range(776, 813)) + list(
                range(849, 887)) + list(range(957, 993)),  # fold7
            list(range(272, 296)) + list(range(671, 707)) + list(range(813, 849)) + list(range(887, 921)) + list(range(921, 957))   # fold8
        ]

    return fold_ranges
