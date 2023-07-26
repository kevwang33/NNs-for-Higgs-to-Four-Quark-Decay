import ROOT
import sys
import numpy as np
import h5py


if len(sys.argv) != 3:
    print ("USAGE:%s<inpput file> <output file>"%(sys.argv[0]))
    sys.exit(1)
    
inFileName = sys.argv[1]
outFileName = sys.argv[2]


print ("Reading from", inFileName, "and writing to", outFileName)

inFile = ROOT.TFile.Open(inFileName ,"READ")
tree = inFile.Get("analysis")

def read_first_entry(vector):
    return vector[0]


jet_label = []
jet_attr = []
jet = []


for entryNum in range(0, tree.GetEntries()):
    tree.GetEntry(entryNum)
    v_jets =[] #list of all the jet four vectors
    v_quarks = [] #list of all the quark four vectors
    g_id = getattr(tree, "g_id")
    pt = getattr(tree, "g_pt")
    for i in range(len(g_id)):
        if g_id[i] ==25: #get all the Higgs Boson four vector
            H_eta = getattr(tree, "g_eta")
            H_phi = getattr(tree, "g_phi")
            H_m = 125.1*10**3 #MeV/c^2
            Higgs = ROOT.TLorentzVector()
            Higgs.SetPtEtaPhiM (pt[i], H_eta[i], H_phi[i], H_m)


        elif abs(g_id[i])<=8: #get all the quark four vectors 
            q_eta = getattr(tree, "g_eta")
            q_phi = getattr(tree, "g_phi")
            q_pt = getattr(tree, "g_pt")
            q_m = 0
            quark = ROOT.TLorentzVector()
            quark.SetPtEtaPhiM(q_pt[i],q_eta[i],q_phi[i],q_m)    
            v_quarks.append(quark)  
    
    #get all the jet four vectors
    fj_eta = getattr(tree, "fj_eta")
    fj_phi = getattr(tree, "fj_phi")
    fj_m = getattr(tree, "fj_m")
    fj_pt = getattr(tree, "fj_pt")
    
    for j in range(len(fj_eta)):
        fjet = ROOT.TLorentzVector()
        fjet.SetPtEtaPhiM (fj_pt[j], fj_eta[j], fj_phi[j], fj_m[j])
        v_jets.append(fjet)
    #get all the consituent for all the jets (including their index) 
    fjc_eta = getattr(tree, "fjc_eta")
    fjc_phi = getattr(tree, "fjc_phi")
    fjc_m = getattr(tree, "fjc_m")
    fjc_pt = getattr(tree, "fjc_pt")
    fjc_ind = getattr(tree, "fjc_ind")
    tot_fjc = []
    
    for i in range(len(fjc_pt)):
        fjc_vector = [fjc_pt[i],fjc_eta[i],fjc_phi[i],fjc_m[i],fjc_ind[i]] #[pt, eta, phi, mass, index]
        tot_fjc.append(fjc_vector)

    for v in v_jets:
        quark_count = 0 
        for q in v_quarks:
            if v.DeltaR(q) <0.9: #counting quarks
                quark_count+=1
        if Higgs.DeltaR(v)>1.0: #id recoil jets
            label_entry = [1,0,0]
        elif Higgs.DeltaR(v)<1.0  and quark_count == 3: #id three quark higgs jets
            label_entry = [0,1,0]
        elif Higgs.DeltaR(v)<1.0 and quark_count ==4: #id four quark higgs jets
            label_entry = [0,0,1]
        else: #id other jets
            label_entry = [0,0,0]
        jet_attr_entry = [v.Pt(),v.Eta(),v.Phi(),v.M()] #create jet attribute arrays and label arrays for each jet
        jet_entry = []
        for fjc in tot_fjc: 
            if fjc[4] == v_jets.index(v): #if the constituent index matches the jet index in the jet lists
                row = [fjc[0],fjc[1],fjc[2],fjc[3]]
                jet_entry.append(row)
            else:
                continue
        tot_fjc = [fjc for fjc in tot_fjc if fjc not in jet_entry] #remove the entries that have already been read 
        if len(jet_entry)<30:
            for i in range(30-len(jet_entry)):
                jet_entry.append([0,0,0,0])
        if len(jet_entry)>=30:
            ranked_fjc = sorted(jet_entry, key = read_first_entry, reverse = True)
            jet_entry = ranked_fjc[:30]
        jet.append(jet_entry)
        jet_attr.append(jet_attr_entry)
        jet_label.append(label_entry)


    
jet = np.array(jet)
jet_attr= np.array(jet_attr)
jet_label=np.array(jet_label)




data_file = h5py.File(outFileName, "w")
dset1 = data_file.create_dataset("jet attributes", data= jet_attr)
dset2 = data_file.create_dataset("jet components", data=jet)
dset3 = data_file.create_dataset("jet labels", data= jet_label)







        
    