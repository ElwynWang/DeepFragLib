fin=open('../../../data/DataBase','r')
lines=fin.readlines()
fin.close()
fout=open('allfasta.fasta','w')
for i in range(len(lines)):
	if (i-12)%9==0:
		fout.write('>'+lines[i].strip()+'\n')
	elif (i-13)%9==0:
		fout.write('%s\n'%lines[i].strip())
fout.close()
