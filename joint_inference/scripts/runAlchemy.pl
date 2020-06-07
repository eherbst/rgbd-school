#!/usr/bin/perl -w

#args: dataset smoutdir; mlndir; scene names; scene indices

my $smoutdir = shift @ARGV;
my $mlndir = shift @ARGV;
my @sceneNames = (shift @ARGV, shift @ARGV);
my @sceneIndices = (shift @ARGV, shift @ARGV);

$smoutdir =~ /^(.*)\/[^\/]+$/;
my $datasetdir = $1;
print "datasetdir $datasetdir\n";
my $sceneNameStr = join(' ', @sceneNames);
my $sceneIndexStr = join(' ', @sceneIndices);

my $cmd = "visualizeAlchemyResults $mlndir 2 $datasetdir $sceneNameStr $sceneIndexStr";
print "running $cmd\n";
`$cmd`;
#use forking to run processes simultaneously
if(fork() == 0) #0 = child
{
	`viewCloudNNormals $smoutdir/fgCompClusters-$sceneNames[0].ply 0 0`;
}
elsif(fork() == 0) #0 = child
{
	`viewCloudNNormals $smoutdir/fgCompClusters-$sceneNames[1].ply 0 0`;
}
elsif(fork() == 0) #0 = child
{
	`intersceneWholeSegMatchsetViewer $smoutdir/sceneNSegsLarge-$sceneNames[0].pnl $smoutdir/sceneNSegsLarge-$sceneNames[1].pnl $smoutdir/selectedCorrs-$sceneNames[0]-$sceneNames[1].dat`;
}
