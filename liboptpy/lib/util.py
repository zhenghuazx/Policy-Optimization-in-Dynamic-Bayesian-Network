
def comupute_cumulative_feed(feed_profile):
    total_feed = 0
    # if isMDP:
    #     total_feed = 0
    #     for i in range(feed_BN.shape[0] - 1):
    #         total_feed += (feed_BN.iloc[i + 1][0] - feed_BN.iloc[i][0]) * (feed_BN.iloc[i][1])
    # else:
    for i, feed_rate in enumerate(feed_profile[1,:-1]):
        total_feed += feed_rate * (feed_profile[0,i+1] - feed_profile[0,i])
    return total_feed

