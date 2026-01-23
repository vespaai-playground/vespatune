# VespaTune

No code tool for training tabular models.

<div align="center">
<a href="https://huggingface.co/spaces/vespa-engine/vespatune?duplicate=true">
<svg width="148" height="20" viewBox="0 0 148 20" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="0.5" y="0.5" width="146.778" height="19" rx="9.5" fill="white" stroke="#EFEFEF"/>
<path d="M27.7778 13.8072V5.68093H29.9522C31.1948 5.68093 32.164 6.01642 32.8598 6.68739C33.5556 7.35837 33.9036 8.36484 33.9036 9.7068C33.9036 11.057 33.5598 12.0801 32.8722 12.7759C32.1847 13.4635 31.2362 13.8072 30.0268 13.8072H27.7778ZM29.2191 12.6392H29.8528C30.6729 12.6392 31.3066 12.4031 31.7539 11.931C32.2013 11.4505 32.4249 10.7091 32.4249 9.7068C32.4249 8.71276 32.2013 7.98793 31.7539 7.53233C31.3066 7.07673 30.6729 6.84893 29.8528 6.84893H29.2191V12.6392Z" fill="#2C3236"/>
<path d="M38.0288 13.9563C37.4655 13.9563 36.9561 13.8321 36.5005 13.5836C36.0449 13.3268 35.6846 12.9623 35.4195 12.4901C35.1544 12.0097 35.0219 11.434 35.0219 10.763C35.0219 10.1003 35.1544 9.5287 35.4195 9.04825C35.6928 8.56779 36.0449 8.19917 36.4756 7.94237C36.9064 7.68558 37.3579 7.55718 37.83 7.55718C38.385 7.55718 38.8489 7.68144 39.2217 7.92995C39.5945 8.17018 39.8761 8.50981 40.0666 8.94884C40.2572 9.37959 40.3524 9.88076 40.3524 10.4523C40.3524 10.7505 40.3317 10.9825 40.2903 11.1482H36.4135C36.4798 11.6866 36.6745 12.1049 36.9975 12.4031C37.3206 12.7014 37.7265 12.8505 38.2152 12.8505C38.4803 12.8505 38.7247 12.8132 38.9483 12.7386C39.1803 12.6558 39.4081 12.544 39.6317 12.4031L40.1163 13.2978C39.8264 13.4883 39.5033 13.6457 39.1471 13.77C38.7909 13.8942 38.4182 13.9563 38.0288 13.9563ZM36.4011 10.179H39.1099C39.1099 9.7068 39.0063 9.33818 38.7992 9.0731C38.5921 8.79974 38.2815 8.66306 37.8673 8.66306C37.5111 8.66306 37.1922 8.79559 36.9105 9.06067C36.6372 9.31747 36.4674 9.69023 36.4011 10.179Z" fill="#2C3236"/>
<path d="M41.7172 16.2178V7.70629H42.8976L42.997 8.35242H43.0467C43.3035 8.13704 43.5893 7.95066 43.9041 7.79327C44.2271 7.63588 44.5544 7.55718 44.8857 7.55718C45.6478 7.55718 46.2359 7.83883 46.6501 8.40212C47.0726 8.96541 47.2838 9.71923 47.2838 10.6636C47.2838 11.3594 47.1596 11.9558 46.9111 12.4528C46.6626 12.9416 46.3395 13.3143 45.9419 13.5711C45.5525 13.8279 45.1301 13.9563 44.6745 13.9563C44.4094 13.9563 44.1443 13.8983 43.8792 13.7824C43.6142 13.6581 43.3574 13.4924 43.1088 13.2854L43.1461 14.3043V16.2178H41.7172ZM44.3763 12.7759C44.7822 12.7759 45.1218 12.5978 45.3951 12.2416C45.6685 11.8854 45.8052 11.3635 45.8052 10.676C45.8052 10.063 45.7016 9.58669 45.4946 9.24706C45.2875 8.90742 44.952 8.73761 44.4881 8.73761C44.0573 8.73761 43.61 8.96541 43.1461 9.42101V12.2665C43.3698 12.4487 43.5852 12.5812 43.7923 12.6641C43.9993 12.7386 44.194 12.7759 44.3763 12.7759Z" fill="#2C3236"/>
<path d="M50.1225 13.9563C49.6172 13.9563 49.2568 13.8031 49.0415 13.4966C48.8344 13.1901 48.7308 12.7718 48.7308 12.2416V5.0348H50.1598V12.3162C50.1598 12.4901 50.1929 12.6144 50.2592 12.6889C50.3254 12.7552 50.3958 12.7883 50.4704 12.7883C50.5035 12.7883 50.5325 12.7883 50.5574 12.7883C50.5905 12.7801 50.6361 12.7718 50.6941 12.7635L50.8804 13.8321C50.7976 13.8652 50.6899 13.8942 50.5574 13.9191C50.4331 13.9439 50.2882 13.9563 50.1225 13.9563Z" fill="#2C3236"/>
<path d="M54.6017 13.9563C54.0964 13.9563 53.6242 13.8321 53.1852 13.5836C52.7462 13.3268 52.39 12.9623 52.1166 12.4901C51.8432 12.0097 51.7065 11.434 51.7065 10.763C51.7065 10.0837 51.8432 9.50799 52.1166 9.03582C52.39 8.55537 52.7462 8.19088 53.1852 7.94237C53.6242 7.68558 54.0964 7.55718 54.6017 7.55718C55.1153 7.55718 55.5916 7.68558 56.0306 7.94237C56.4697 8.19088 56.8259 8.55537 57.0992 9.03582C57.3726 9.50799 57.5093 10.0837 57.5093 10.763C57.5093 11.434 57.3726 12.0097 57.0992 12.4901C56.8259 12.9623 56.4697 13.3268 56.0306 13.5836C55.5916 13.8321 55.1153 13.9563 54.6017 13.9563ZM54.6017 12.7883C55.049 12.7883 55.4011 12.602 55.6579 12.2292C55.9147 11.8564 56.0431 11.3677 56.0431 10.763C56.0431 10.15 55.9147 9.6571 55.6579 9.28433C55.4011 8.91157 55.049 8.72518 54.6017 8.72518C54.1544 8.72518 53.8023 8.91157 53.5455 9.28433C53.297 9.6571 53.1728 10.15 53.1728 10.763C53.1728 11.3677 53.297 11.8564 53.5455 12.2292C53.8023 12.602 54.1544 12.7883 54.6017 12.7883Z" fill="#2C3236"/>
<path d="M59.2609 16.3172C59.0952 16.3172 58.9502 16.3048 58.826 16.2799C58.7017 16.2551 58.5816 16.2261 58.4657 16.1929L58.7266 15.0746C58.7846 15.0912 58.8508 15.1078 58.9254 15.1243C59.0082 15.1492 59.0869 15.1616 59.1615 15.1616C59.4845 15.1616 59.7413 15.0622 59.9319 14.8634C60.1307 14.6729 60.2798 14.4244 60.3792 14.1179L60.491 13.7327L58.0929 7.70629H59.5467L60.6028 10.7008C60.694 10.9659 60.7809 11.2476 60.8638 11.5458C60.9549 11.8357 61.046 12.1256 61.1371 12.4156H61.1868C61.2614 12.1339 61.336 11.8481 61.4105 11.5582C61.4933 11.26 61.572 10.9742 61.6466 10.7008L62.5661 7.70629H63.9453L61.7336 14.0806C61.4768 14.7681 61.1661 15.3107 60.8017 15.7083C60.4372 16.1142 59.9236 16.3172 59.2609 16.3172Z" fill="#2C3236"/>
<path d="M69.988 13.9563C69.4827 13.9563 69.0105 13.8321 68.5715 13.5836C68.1325 13.3268 67.7763 12.9623 67.5029 12.4901C67.2295 12.0097 67.0929 11.434 67.0929 10.763C67.0929 10.0837 67.2295 9.50799 67.5029 9.03582C67.7763 8.55537 68.1325 8.19088 68.5715 7.94237C69.0105 7.68558 69.4827 7.55718 69.988 7.55718C70.5016 7.55718 70.9779 7.68558 71.4169 7.94237C71.856 8.19088 72.2122 8.55537 72.4855 9.03582C72.7589 9.50799 72.8956 10.0837 72.8956 10.763C72.8956 11.434 72.7589 12.0097 72.4855 12.4901C72.2122 12.9623 71.856 13.3268 71.4169 13.5836C70.9779 13.8321 70.5016 13.9563 69.988 13.9563ZM69.988 12.7883C70.4353 12.7883 70.7874 12.602 71.0442 12.2292C71.301 11.8564 71.4294 11.3677 71.4294 10.763C71.4294 10.15 71.301 9.6571 71.0442 9.28433C70.7874 8.91157 70.4353 8.72518 69.988 8.72518C69.5407 8.72518 69.1886 8.91157 68.9318 9.28433C68.6833 9.6571 68.5591 10.15 68.5591 10.763C68.5591 11.3677 68.6833 11.8564 68.9318 12.2292C69.1886 12.602 69.5407 12.7883 69.988 12.7883Z" fill="#2C3236"/>
<path d="M74.31 13.8072V7.70629H75.4904L75.5898 8.52637H75.6395C75.9129 8.2613 76.2111 8.03349 76.5341 7.84297C76.8572 7.65244 77.2258 7.55718 77.64 7.55718C78.2944 7.55718 78.7707 7.76842 79.0689 8.19088C79.3672 8.61335 79.5163 9.20978 79.5163 9.98016V13.8072H78.0873V10.1665C78.0873 9.66124 78.0128 9.30504 77.8637 9.09795C77.7146 8.89086 77.4702 8.78731 77.1306 8.78731C76.8655 8.78731 76.6294 8.85358 76.4223 8.98612C76.2235 9.11037 75.9957 9.29676 75.7389 9.54527V13.8072H74.31Z" fill="#2C3236"/>
<path d="M83.8747 13.8072V5.68093H85.3161V8.94884H88.6088V5.68093H90.0502V13.8072H88.6088V10.2038H85.3161V13.8072H83.8747Z" fill="#2C3236"/>
<path d="M92.1139 13.8072V5.68093H97.022V6.89863H93.5553V9.19735H96.5126V10.4151H93.5553V13.8072H92.1139Z" fill="#2C3236"/>
<path d="M103.309 13.9563C102.762 13.9563 102.232 13.8528 101.718 13.6457C101.213 13.4386 100.766 13.1445 100.376 12.7635L101.221 11.7819C101.511 12.0552 101.843 12.2789 102.215 12.4528C102.588 12.6185 102.961 12.7014 103.334 12.7014C103.798 12.7014 104.15 12.6061 104.39 12.4156C104.63 12.225 104.75 11.9724 104.75 11.6576C104.75 11.318 104.63 11.0736 104.39 10.9245C104.158 10.7754 103.86 10.6221 103.495 10.4648L102.377 9.98016C102.112 9.86419 101.851 9.71508 101.594 9.53284C101.346 9.3506 101.139 9.11866 100.973 8.83701C100.815 8.55537 100.737 8.21574 100.737 7.81812C100.737 7.38737 100.853 7.00217 101.085 6.66254C101.325 6.31463 101.648 6.04127 102.054 5.84246C102.468 5.63537 102.94 5.53182 103.47 5.53182C103.943 5.53182 104.398 5.62708 104.837 5.81761C105.276 5.99985 105.653 6.24836 105.968 6.56314L105.235 7.48263C104.978 7.26725 104.705 7.09744 104.415 6.97318C104.133 6.84893 103.818 6.7868 103.47 6.7868C103.089 6.7868 102.779 6.87378 102.538 7.04774C102.307 7.21341 102.191 7.44535 102.191 7.74357C102.191 7.95066 102.249 8.12462 102.365 8.26544C102.489 8.39798 102.65 8.51395 102.849 8.61335C103.048 8.70447 103.259 8.79559 103.483 8.88672L104.589 9.34646C105.069 9.55355 105.463 9.82691 105.769 10.1665C106.076 10.4979 106.229 10.9576 106.229 11.5458C106.229 11.9848 106.113 12.3866 105.881 12.7511C105.649 13.1155 105.313 13.4096 104.874 13.6333C104.444 13.8486 103.922 13.9563 103.309 13.9563Z" fill="#2C3236"/>
<path d="M107.582 16.2178V7.70629H108.763L108.862 8.35242H108.912C109.169 8.13704 109.454 7.95066 109.769 7.79327C110.092 7.63588 110.419 7.55718 110.751 7.55718C111.513 7.55718 112.101 7.83883 112.515 8.40212C112.938 8.96541 113.149 9.71923 113.149 10.6636C113.149 11.3594 113.025 11.9558 112.776 12.4528C112.528 12.9416 112.205 13.3143 111.807 13.5711C111.418 13.8279 110.995 13.9563 110.54 13.9563C110.274 13.9563 110.009 13.8983 109.744 13.7824C109.479 13.6581 109.222 13.4924 108.974 13.2854L109.011 14.3043V16.2178H107.582ZM110.241 12.7759C110.647 12.7759 110.987 12.5978 111.26 12.2416C111.534 11.8854 111.67 11.3635 111.67 10.676C111.67 10.063 111.567 9.58669 111.36 9.24706C111.153 8.90742 110.817 8.73761 110.353 8.73761C109.922 8.73761 109.475 8.96541 109.011 9.42101V12.2665C109.235 12.4487 109.45 12.5812 109.657 12.6641C109.864 12.7386 110.059 12.7759 110.241 12.7759Z" fill="#2C3236"/>
<path d="M115.929 13.9563C115.399 13.9563 114.964 13.7907 114.624 13.4593C114.293 13.128 114.127 12.6972 114.127 12.1671C114.127 11.5126 114.413 11.0073 114.985 10.6511C115.556 10.2867 116.467 10.0381 117.718 9.90561C117.71 9.58255 117.623 9.30504 117.457 9.0731C117.3 8.83287 117.014 8.71276 116.6 8.71276C116.302 8.71276 116.008 8.77074 115.718 8.88672C115.436 9.00269 115.159 9.14351 114.885 9.30918L114.363 8.35242C114.703 8.13704 115.084 7.95066 115.506 7.79327C115.937 7.63588 116.393 7.55718 116.873 7.55718C117.635 7.55718 118.203 7.78498 118.576 8.24059C118.957 8.68791 119.147 9.33818 119.147 10.1914V13.8072H117.967L117.867 13.1362H117.818C117.544 13.3682 117.25 13.5629 116.935 13.7203C116.629 13.8776 116.293 13.9563 115.929 13.9563ZM116.389 12.838C116.637 12.838 116.861 12.7801 117.06 12.6641C117.267 12.5398 117.486 12.3741 117.718 12.1671V10.8002C116.89 10.9079 116.314 11.0695 115.991 11.2848C115.668 11.4919 115.506 11.7487 115.506 12.0552C115.506 12.3286 115.589 12.5274 115.755 12.6517C115.921 12.7759 116.132 12.838 116.389 12.838Z" fill="#2C3236"/>
<path d="M123.404 13.9563C122.849 13.9563 122.348 13.8321 121.901 13.5836C121.462 13.3268 121.11 12.9623 120.845 12.4901C120.588 12.0097 120.46 11.434 120.46 10.763C120.46 10.0837 120.6 9.50799 120.882 9.03582C121.164 8.55537 121.536 8.19088 122 7.94237C122.473 7.68558 122.978 7.55718 123.516 7.55718C123.906 7.55718 124.245 7.62345 124.535 7.75599C124.825 7.88853 125.082 8.05006 125.306 8.24059L124.61 9.1725C124.452 9.03168 124.291 8.92399 124.125 8.84944C123.959 8.7666 123.781 8.72518 123.591 8.72518C123.102 8.72518 122.7 8.91157 122.386 9.28433C122.079 9.6571 121.926 10.15 121.926 10.763C121.926 11.3677 122.075 11.8564 122.373 12.2292C122.68 12.602 123.073 12.7883 123.554 12.7883C123.794 12.7883 124.017 12.7386 124.225 12.6392C124.44 12.5315 124.635 12.4073 124.809 12.2665L125.393 13.2108C125.111 13.4593 124.796 13.6457 124.448 13.77C124.1 13.8942 123.752 13.9563 123.404 13.9563Z" fill="#2C3236"/>
<path d="M128.903 13.9563C128.339 13.9563 127.83 13.8321 127.374 13.5836C126.919 13.3268 126.558 12.9623 126.293 12.4901C126.028 12.0097 125.896 11.434 125.896 10.763C125.896 10.1003 126.028 9.5287 126.293 9.04825C126.567 8.56779 126.919 8.19917 127.35 7.94237C127.78 7.68558 128.232 7.55718 128.704 7.55718C129.259 7.55718 129.723 7.68144 130.096 7.92995C130.468 8.17018 130.75 8.50981 130.941 8.94884C131.131 9.37959 131.226 9.88076 131.226 10.4523C131.226 10.7505 131.206 10.9825 131.164 11.1482H127.287C127.354 11.6866 127.548 12.1049 127.871 12.4031C128.194 12.7014 128.6 12.8505 129.089 12.8505C129.354 12.8505 129.599 12.8132 129.822 12.7386C130.054 12.6558 130.282 12.544 130.506 12.4031L130.99 13.2978C130.7 13.4883 130.377 13.6457 130.021 13.77C129.665 13.8942 129.292 13.9563 128.903 13.9563ZM127.275 10.179H129.984C129.984 9.7068 129.88 9.33818 129.673 9.0731C129.466 8.79974 129.155 8.66306 128.741 8.66306C128.385 8.66306 128.066 8.79559 127.784 9.06067C127.511 9.31747 127.341 9.69023 127.275 10.179Z" fill="#2C3236"/>
<path d="M134.293 13.9563C133.879 13.9563 133.465 13.8776 133.051 13.7203C132.637 13.5546 132.28 13.3516 131.982 13.1114L132.653 12.1919C132.927 12.399 133.2 12.5647 133.473 12.6889C133.747 12.8132 134.037 12.8753 134.343 12.8753C134.674 12.8753 134.919 12.8049 135.076 12.6641C135.234 12.5233 135.312 12.3493 135.312 12.1422C135.312 11.9682 135.246 11.8274 135.113 11.7197C134.989 11.6038 134.828 11.5044 134.629 11.4215C134.43 11.3304 134.223 11.2434 134.008 11.1606C133.743 11.0612 133.477 10.9411 133.212 10.8002C132.956 10.6511 132.744 10.4648 132.579 10.2411C132.413 10.0092 132.33 9.72337 132.33 9.38374C132.33 8.8453 132.529 8.40626 132.927 8.06663C133.324 7.727 133.863 7.55718 134.542 7.55718C134.973 7.55718 135.358 7.63174 135.697 7.78084C136.037 7.92995 136.331 8.09976 136.58 8.29029L135.921 9.16008C135.706 9.00269 135.486 8.87843 135.263 8.78731C135.047 8.68791 134.819 8.6382 134.579 8.6382C134.273 8.6382 134.045 8.70447 133.896 8.83701C133.747 8.96127 133.672 9.11866 133.672 9.30918C133.672 9.54941 133.796 9.73165 134.045 9.85591C134.293 9.98016 134.583 10.1003 134.915 10.2162C135.196 10.3157 135.47 10.4399 135.735 10.589C136 10.7298 136.219 10.9162 136.393 11.1482C136.576 11.3801 136.667 11.6866 136.667 12.0677C136.667 12.5895 136.464 13.0368 136.058 13.4096C135.652 13.7741 135.064 13.9563 134.293 13.9563Z" fill="#2C3236"/>
<path d="M13.5903 10.8519V12.8074H15.5458V10.8519H13.5903Z" fill="#FF3270"/>
<path d="M17.5186 10.8519V12.8074H19.474V10.8519H17.5186Z" fill="#861FFF"/>
<path d="M13.5903 6.92394V8.87943H15.5458V6.92394H13.5903Z" fill="#097EFF"/>
<path fill-rule="evenodd" clip-rule="evenodd" d="M12.2222 6.42394C12.2222 5.94434 12.611 5.55556 13.0906 5.55556H16.0458C16.447 5.55556 16.7846 5.82753 16.8843 6.19714C17.3046 5.79944 17.872 5.55556 18.4963 5.55556C19.792 5.55556 20.8424 6.6059 20.8424 7.90154C20.8424 8.52588 20.5985 9.09326 20.2007 9.5136C20.5704 9.61334 20.8424 9.95092 20.8424 10.3521V13.3073C20.8424 13.7869 20.4535 14.1757 19.9739 14.1757H16.9142H16.1503H13.0906C12.611 14.1757 12.2222 13.7869 12.2222 13.3073V9.56341V9.48368V6.42394ZM13.5905 6.92382V8.8793H15.546V6.92382H13.5905ZM13.5905 12.8074V10.8519H15.546V12.8074H13.5905ZM17.5186 12.8074V10.8519H19.4741V12.8074H17.5186ZM17.5186 7.90154C17.5186 7.36156 17.9564 6.92382 18.4963 6.92382C19.0363 6.92382 19.4741 7.36156 19.4741 7.90154C19.4741 8.44155 19.0363 8.8793 18.4963 8.8793C17.9564 8.8793 17.5186 8.44155 17.5186 7.90154Z" fill="black"/>
<path d="M18.4963 6.92394C17.9563 6.92394 17.5186 7.36169 17.5186 7.9017C17.5186 8.44167 17.9563 8.87943 18.4963 8.87943C19.0363 8.87943 19.474 8.44167 19.474 7.9017C19.474 7.36169 19.0363 6.92394 18.4963 6.92394Z" fill="#FFD702"/>
</svg>
</a>
</div>

- **Web UI** for training, monitoring, and managing models
- Tune models directly from CSV files
- Real-time training progress with WebSocket updates
- Export models to ONNX format for deployment

## Installation

Install using pip:

```bash
pip install vespatune
```

## Quick Start

### Web UI (Recommended)

Start the web interface:

```bash
vespatune
```

This launches the VespaTune UI at `http://127.0.0.1:9999` where you can:
- Upload train/validation CSV files
- Configure model type, target columns, and hyperparameters
- Start training with real-time progress monitoring
- View trial results and metrics
- Download trained models and artifacts
- Manage multiple training runs

You can also specify host and port:

```bash
vespatune --host 0.0.0.0 --port 8080
```

### CLI

Train a model with explicit train/valid split:

```bash
vespatune train \
  --train_filename train.csv \
  --valid_filename valid.csv \
  --output outputs/my_model \
  --model xgboost
```

Or let VespaTune auto-split your data:

```bash
vespatune train \
  --train_filename data.csv \
  --output outputs/my_model \
  --model xgboost
```

Make predictions:

```bash
vespatune predict \
  --model_path outputs/my_model \
  --test_filename test.csv \
  --output_filename predictions.csv
```

Serve a trained model for predictions:

```bash
vespatune serve --model_path outputs/my_model --host 0.0.0.0 --port 8000
```

### Python API

```python
from vespatune import VespaTune

# With explicit validation file
vtune = VespaTune(
    train_filename="train.csv",
    valid_filename="valid.csv",
    output="outputs/my_model",
    model_type="xgboost",  # or "lightgbm" or "catboost"
    targets=["target"],
    num_trials=100,
    time_limit=3600,
)
vtune.train()

# Or with auto-split (no validation file needed)
vtune = VespaTune(
    train_filename="data.csv",
    output="outputs/my_model",
    model_type="xgboost",
    targets=["target"],
    num_trials=100,
)
vtune.train()
```

## Web UI Features

The web interface provides:

- **File Upload**: Drag and drop CSV files for training (validation file is optional)
- **Auto-Split**: If no validation file is provided, automatically splits training data
- **Auto Column Detection**: Automatically detects columns for target and ID selection
- **Model Selection**: Choose between XGBoost, LightGBM, or CatBoost
- **Real-time Monitoring**: Watch training progress with live trial updates via WebSocket
- **Metrics Visualization**: View loss curves and hyperparameter importance
- **Run Management**: Start, stop, and delete training runs
- **Artifact Downloads**: Download trained models, configs, and ONNX exports

## Parameters

### Required

| Parameter | Description |
|-----------|-------------|
| `train_filename` | Path to training CSV file |
| `output` | Path to output directory for model artifacts |

### Optional

| Parameter | Default | Description |
|-----------|---------|-------------|
| `valid_filename` | `None` | Path to validation CSV file (auto-splits training data if not provided) |
| `model_type` | `"xgboost"` | Model to use: `"xgboost"`, `"lightgbm"`, `"catboost"`, or `"logreg"` |
| `test_filename` | `None` | Path to test CSV file (predictions saved if provided) |
| `task` | `None` | `"classification"` or `"regression"` (auto-detected if not specified) |
| `idx` | `"id"` | Name of the ID column |
| `targets` | `["target"]` | List of target column names |
| `features` | `None` | List of feature columns (all non-id/target columns if not specified) |
| `categorical_features` | `None` | List of categorical columns (auto-detected if not specified) |
| `use_gpu` | `False` | Whether to use GPU for training |
| `seed` | `42` | Random seed for reproducibility |
| `num_trials` | `1000` | Number of Optuna trials for hyperparameter tuning |
| `time_limit` | `None` | Time limit for optimization in seconds |

## Supported Models

### XGBoost
- Default model with extensive hyperparameter search
- Supports GPU acceleration
- Best for general-purpose tasks

### LightGBM
- Native categorical feature support
- Fast training on large datasets
- Supports GPU acceleration

### CatBoost
- Best native categorical feature handling
- Robust to overfitting
- Supports GPU acceleration

### Logistic Regression
- Linear model for classification tasks only
- Searches over preprocessing (imputation, scaling) and regularization
- Fast training, interpretable coefficients

## Data Splitting

VespaTune supports two modes:

1. **Explicit split**: Provide both `train_filename` and `valid_filename`
2. **Auto-split**: Provide only `train_filename` - VespaTune automatically creates a 5-fold split and uses fold 0 (80% train, 20% valid)

For manual control over splits, use the splitter utility:

```bash
vespatune splitter \
  --data_filename data.csv \
  --output splits/ \
  --target target \
  --task classification \
  --num_folds 5
```

Or via Python:

```python
from vespatune import VespaTuneSplitter

splitter = VespaTuneSplitter(
    data_filename="data.csv",
    output="splits/",
    target="target",
    task="classification",
    num_folds=5,
)
splitter.split()
```

This creates `fold_0_train.csv`, `fold_0_valid.csv`, etc. for k-fold cross-validation.


## Prediction

### Using the trained model

```python
from vespatune import VespaTunePredict

predictor = VespaTunePredict(model_path="outputs/my_model")

# Predict on file
predictor.predict_file("test.csv", "predictions.csv")

# Predict single sample
prediction = predictor.predict_single({"feature1": 1.0, "feature2": "A"})
```

### Using ONNX model

```python
from vespatune import VespaTuneONNXPredict

predictor = VespaTuneONNXPredict(model_path="onnx_model/")

# Predict on file
predictor.predict_file("test.csv", "predictions.csv")

# Predict single sample
prediction = predictor.predict_single({"feature1": 1.0, "feature2": "A"})
```

### Standalone Preprocessing

Use `VespaTuneProcessor` when you want to preprocess data independently and pass it to an external ONNX runtime or inference system:

```python
from vespatune import VespaTuneProcessor
import onnxruntime as ort

# Load preprocessor from model or ONNX export directory
processor = VespaTuneProcessor(model_path="outputs/my_model")

# Transform DataFrame
processed = processor.transform(df)  # Returns float32 numpy array

# Transform single sample
processed = processor.transform_single({"feature1": 1.0, "feature2": "A"})

# Get feature metadata
processor.get_feature_names()        # Input feature names
processor.get_categorical_features() # Categorical feature names
processor.get_feature_names_out()    # Output feature names after transform
processor.get_input_schema()         # Pydantic schema for API validation

# Pass to ONNX runtime
session = ort.InferenceSession("model.onnx")
predictions = session.run(None, {"input": processed})
```

## CLI Reference

### Default (UI)

```bash
vespatune [--host HOST] [--port PORT]

options:
  --host                Host to serve on (default: 127.0.0.1)
  --port                Port to serve on (default: 9999)
  --version, -v         Display VespaTune version
```

### train

```bash
vespatune train --help

options:
  --train_filename      Path to training file (required)
  --valid_filename      Path to validation file (optional, auto-splits if not provided)
  --output              Path to output directory (required)
  --model               Model type: xgboost, lightgbm, catboost, logreg (default: xgboost)
  --test_filename       Path to test file
  --task                Task type: classification, regression
  --idx                 ID column name
  --targets             Target column(s), separate multiple by ';'
  --features            Feature columns, separate by ';'
  --use_gpu             Use GPU for training
  --seed                Random seed (default: 42)
  --num_trials          Number of Optuna trials (default: 100)
  --time_limit          Time limit in seconds
```

### predict

```bash
vespatune predict --help

options:
  --model_path          Path to trained model directory (required)
  --test_filename       Path to test file (required)
  --output_filename     Path to output predictions file (required)
```

### export

```bash
vespatune export --help

options:
  --model_path          Path to trained model directory (required)
  --output_dir          Path to ONNX output directory
```

### serve

```bash
vespatune serve --help

options:
  --model_path          Path to ONNX export directory
  --host                Host to bind (default: 127.0.0.1)
  --port                Port to bind (default: 9999)
  --workers             Number of workers (default: 1)
  --reload              Enable auto-reload for development
```

### splitter

```bash
vespatune splitter --help

options:
  --data_filename       Path to data file (required)
  --output              Path to output directory (required)
  --target              Target column name (required)
  --task                Task type: classification, regression (required)
  --num_folds           Number of folds (default: 5)
```

## Output Files

After training, the following files are created in the output directory:

| File | Description |
|------|-------------|
| `vtune_model.final` | Trained model |
| `vtune.config` | Model configuration |
| `vtune.best_params` | Best hyperparameters from Optuna |
| `vtune.preprocessor.joblib` | Fitted preprocessor (encoding, scaling, imputation) |
| `vtune.target_encoder` | Target encoder (for classification) |
| `params.db` | Optuna study database |
| `train.feather` | Processed training data |
| `valid.feather` | Processed validation data |
| `onnx/` | ONNX export directory (after export) |
| `_splits/` | Auto-generated train/valid splits (only if no validation file provided) |

## Example

```python
from vespatune import VespaTune

# Train with LightGBM
vtune = VespaTune(
    train_filename="data/train.csv",
    valid_filename="data/valid.csv",
    output="outputs/lgb_model",
    model_type="lightgbm",
    targets=["price"],
    task="regression",
    num_trials=200,
    time_limit=1800,
    use_gpu=False,
    seed=42,
)
vtune.train()
```
