Dataset: Four class motor imagery(001-2014) http://bnci-horizon-2020.eu/database/data-sets
 
I see. This is a popular dataset, but I should say I am not sure it is the best one.
Maybe you could consider another one, which is very high quality, has many participants and more EEG channels (64).

It can be downloaded from here https://physionet.org/content/eegmmidb/1.0.0/
I have used it before, and obtained good results easily.
 
Tutorial: Trialwise Decoding on BCIC IV 2a Dataset: https://braindecode.org/auto_examples/plot_bcic_iv_2a_moabb_trial.html
 
I see. Then, actually you do seem to preprocess your data, as detailed in the section "Preprocessing" under that link.
When using the method below, please skip this step as it's already done.
 
Now, in the below I have prepared some code especially for you.
It's really easy to use and does all steps of data augmentation for the dataset you are using now.
You can run it easily using MatLab, then continue with the rest in Python as usual.
 
Please, could you try it and let me know if you can get good results?
I am really interested, and grateful for your time.

```
clear
load('A01T.mat'); ← this has to be updated with actual name, of course
 
f=10; ← frequency to emphasize, please try changing maybe 8~16 in steps of 2 Hz
k=0.6; ← coupling parameter. The higher, the more similar the input and output signals. Please try maybe 0.2~0.8 in steps of 0.1
a=0.3; ← chaos parameter. The best choice needs to be found, please try chaging maybe 0.15~0.3 in steps of 0.05
b=0.2; ← don't change, please
c=5.7; ← don't change, please
 
for ii=1:length(data)
   eeg=data{ii}.X'; ← extract original EEG
   fs=data{ii}.fs;
   n=size(eeg,1);
   
   feeg=zeros(size(eeg)); ←run filtering, normalization
   eeg=eeg-repmat(mean(eeg),n,1);
   [fb,fa]=butter(2,[7,30]/(fs/2));
   for j=1:n
       feeg(j,:)=filter(fb,fa,eeg(j,:)-mean(eeg(j,:)));
       feeg(j,:)=feeg(j,:)/std(feeg(j,:));
   end
   
   aeeg=zeros(size(eeg)); ←perform data augmentation
   parfor j=1:n
       ts=feeg(j,:);
       tr=(0:length(feeg)-1)/fs;
       cp=2;
       ts_nlf=cp*tanh(ts/cp);
       tid=0.1/fs;
       ti=0:tid:max(tr);
       ts_nlfi=interp1(tr,ts_nlf,ti,'pchip');
       fun=@(t,y) 2*pi*f*[-y(2)-y(3)+(ts_nlfi(round(t/tid)+1)-y(1))*k;
           y(1)+a*y(2);
           b+y(3)*(y(1)-c)];
       y0=[6,0,0];
       [t,y]=ode45(fun,ti,y0);
       aeeg(j,:)=y(1:10:end,1)-mean(y(:,1));
   end
   
   data{ii}.X=[feeg;aeeg]'; ←assemble output EEG
end
 
save('A01T_augmented.mat','data'); ← this has to be updated with actual name, of course

```
 
You'll find that instead of 25 channels, you obtain 50. 
The last 25 are the output from the system.
Of course you should remove those that did not actually correspond to EEG signals but other signals such as EOG.
 
As there may be too many combinations of the parametrs 'a', 'k' and 'f', actually you can try changing one parameter at a time. 
For example, start by finding which one has the biggest effect, such as, first change k, then change f then change a. 
Decide which one has the biggest effect, set that parameter to the best value. 
Then repeat for the others.
Of course a full search grid is much better, if it is not too slow.
 
Maybe you could first of all try to see what happens on your best participant, just as an informal starting point.
